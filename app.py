

import os
import io
import base64
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from PIL import Image, UnidentifiedImageError
import numpy as np

from keras.models import load_model

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("airvisionnet")

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = Flask(__name__)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_ORIGINS_ENV = os.environ.get("ALLOWED_ORIGINS", "*")
if ALLOWED_ORIGINS_ENV == "*":
    _cors_origins = "*"
    logger.warning(
        "ALLOWED_ORIGINS is not set — CORS is wide open ('*'). "
        "Fine for local development; set ALLOWED_ORIGINS to a comma-separated "
        "allow-list (e.g. 'https://yourapp.com') before deploying publicly."
    )
else:
    _cors_origins = [o.strip() for o in ALLOWED_ORIGINS_ENV.split(",") if o.strip()]
CORS(app, resources={r"/api/*": {"origins": _cors_origins}})

API_KEY = os.environ.get("API_KEY")  # if unset, mutating routes are unprotected (dev mode)
if not API_KEY:
    logger.warning(
        "API_KEY is not set — destructive endpoints (DELETE /api/records/<id>) "
        "are unprotected. Set API_KEY before deploying publicly."
    )

# Optional rate limiting (soft dependency)
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    limiter = Limiter(get_remote_address, app=app, default_limits=[])
    logger.info("flask-limiter available — rate limiting enabled on /api/predict.")
except ImportError:
    limiter = None
    logger.info(
        "flask-limiter not installed — /api/predict has no rate limit. "
        "Install flask-limiter to enable it."
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "predictions.db")


def rate_limited(rule):
    """No-op decorator when flask-limiter isn't installed."""
    def decorator(f):
        return limiter.limit(rule)(f) if limiter else f
    return decorator


def require_api_key(f):
    """Protect mutating endpoints. No-op if API_KEY isn't configured."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if API_KEY and request.headers.get("X-API-Key") != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper


# ─────────────────────────────────────────────
# Database (SQLite, WAL mode, indexed, context-managed)
# ─────────────────────────────────────────────
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT NOT NULL,
                filename     TEXT,
                model_used   TEXT,
                pm25         REAL,
                pm10         REAL,
                aqi_pm25     INTEGER,
                aqi_pm10     INTEGER,
                aqi_overall  INTEGER,
                aqi_category TEXT,
                image_b64    TEXT,
                latitude     REAL,
                longitude    REAL,
                is_hotspot   INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_category ON predictions(aqi_category)")
    migrate_db()


def migrate_db():
    """Add any columns missing from an older database file."""
    with get_db() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
        if "latitude" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN latitude REAL")
        if "longitude" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN longitude REAL")
        if "is_hotspot" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN is_hotspot INTEGER DEFAULT 0")
        if "model_used" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN model_used TEXT")


init_db()

# ─────────────────────────────────────────────
# AQI computation – Government of India / CPCB norms
# (unchanged — verified correct against published CPCB breakpoints)
# ─────────────────────────────────────────────
PM25_BREAKPOINTS = [
    # (C_lo, C_hi, AQI_lo, AQI_hi, category)
    (0.0, 30.0, 0, 50, "Good"),
    (30.0, 60.0, 51, 100, "Satisfactory"),
    (60.0, 90.0, 101, 200, "Moderate"),
    (90.0, 120.0, 201, 300, "Poor"),
    (120.0, 250.0, 301, 400, "Very Poor"),
    (250.0, 500.0, 401, 500, "Severe"),
]

PM10_BREAKPOINTS = [
    (0.0, 50.0, 0, 50, "Good"),
    (50.0, 100.0, 51, 100, "Satisfactory"),
    (100.0, 250.0, 101, 200, "Moderate"),
    (250.0, 350.0, 201, 300, "Poor"),
    (350.0, 430.0, 301, 400, "Very Poor"),
    (430.0, 600.0, 401, 500, "Severe"),
]

HOTSPOT_CATEGORIES = {"Poor", "Very Poor", "Severe"}
DEFAULT_MAP_CENTER = {"lat": 23.5204, "lng": 87.3119}  # Durgapur reference


def compute_sub_aqi(concentration, breakpoints):
    for (c_lo, c_hi, aqi_lo, aqi_hi, _) in breakpoints:
        if c_lo <= concentration <= c_hi:
            return int(round(((aqi_hi - aqi_lo) / (c_hi - c_lo)) * (concentration - c_lo) + aqi_lo))
    if concentration > breakpoints[-1][1]:
        return 500
    return 0


def category_from_aqi(aqi_val):
    if aqi_val <= 50:
        return "Good"
    if aqi_val <= 100:
        return "Satisfactory"
    if aqi_val <= 200:
        return "Moderate"
    if aqi_val <= 300:
        return "Poor"
    if aqi_val <= 400:
        return "Very Poor"
    return "Severe"


def aqi_color(category):
    return {
        "Good": "#00b050",
        "Satisfactory": "#92d050",
        "Moderate": "#ffff00",
        "Poor": "#ff0000",
        "Very Poor": "#7030a0",
        "Severe": "#7b0000",
    }.get(category, "#888")


def compute_aqi(pm25, pm10):
    aqi25 = compute_sub_aqi(pm25, PM25_BREAKPOINTS)
    aqi10 = compute_sub_aqi(pm10, PM10_BREAKPOINTS)
    overall = max(aqi25, aqi10)
    category = category_from_aqi(overall)
    return aqi25, aqi10, overall, category, aqi_color(category)


def is_hotspot_prediction(category, latitude, longitude):
    if latitude is None or longitude is None:
        return False
    return category in HOTSPOT_CATEGORIES


def parse_coords(value, lo, hi):
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if lo <= v <= hi else None


# ─────────────────────────────────────────────
# Model loading
#
# Both .h5 files embed their own model_config (verified: keras_version
# 2.10.0, full model_weights + training_config groups present), so we load
# them as complete models rather than rebuilding the architecture in Python
# and loading weights by name. This means a real architecture mismatch
# between the two files raises immediately instead of failing silently.
# ─────────────────────────────────────────────
DAY_MODEL = None
NIGHT_MODEL = None
MODEL_LOADED = False
MODEL_ERROR = None
MODEL_LOADED_AT = None

DAY_MODEL_PATH = os.path.join(BASE_DIR, "LightairnetMainModel.h5")
NIGHT_MODEL_PATH = os.path.join(BASE_DIR, "MainModel_2.h5")
DAY_MODEL_NAME = os.path.basename(DAY_MODEL_PATH)
NIGHT_MODEL_NAME = os.path.basename(NIGHT_MODEL_PATH)

INFERENCE_SIZE = (1024, 1024)

# Day/night heuristic tuning (env-overridable)
DAY_START_HOUR = int(os.environ.get("DAY_START_HOUR", "6"))
NIGHT_START_HOUR = int(os.environ.get("NIGHT_START_HOUR", "18"))
NIGHT_BRIGHTNESS_THRESHOLD = float(os.environ.get("NIGHT_BRIGHTNESS_THRESHOLD", "80"))


def load_models():
    global DAY_MODEL, NIGHT_MODEL, MODEL_LOADED, MODEL_ERROR, MODEL_LOADED_AT

    try:
        logger.info("Loading day model from %s", DAY_MODEL_PATH)
        DAY_MODEL = load_model(DAY_MODEL_PATH, compile=False)

        logger.info("Loading night model from %s", NIGHT_MODEL_PATH)
        NIGHT_MODEL = load_model(NIGHT_MODEL_PATH, compile=False)

        # Warm up both graphs so the first real user request isn't the one
        # that pays for TensorFlow's lazy graph-tracing cost.
        dummy = np.zeros((1, INFERENCE_SIZE[0], INFERENCE_SIZE[1], 3), dtype=np.float32)
        DAY_MODEL.predict(dummy, verbose=0)
        NIGHT_MODEL.predict(dummy, verbose=0)

        MODEL_LOADED = True
        MODEL_ERROR = None
        MODEL_LOADED_AT = datetime.utcnow().isoformat()
        logger.info("Both models loaded and warmed up successfully.")

    except Exception as e:
        DAY_MODEL = None
        NIGHT_MODEL = None
        MODEL_LOADED = False
        MODEL_ERROR = str(e)
        logger.exception("Model loading failed")


load_models()


# ─────────────────────────────────────────────
# Ground-truth extraction (internal test-dataset filename convention)
# Filenames look like: <DD-Mon-YYYY_HH_MM_SS>@<pm10>@<pm25>.<ext>
# ─────────────────────────────────────────────
import re

_GT_PATTERN = re.compile(r"^(.*?)@([\d.]+)@([\d.]+)\.[^.]+$")


def extract_ground_truth(filename):
    match = _GT_PATTERN.match(filename or "")
    if not match:
        return {"capture_time": None, "actual_pm10": None, "actual_pm25": None}
    try:
        capture_time = datetime.strptime(match.group(1), "%d-%b-%Y_%H_%M_%S")
        return {
            "capture_time": capture_time,
            "actual_pm25": float(match.group(2)),
            "actual_pm10": float(match.group(3)),
        }
    except ValueError:
        return {"capture_time": None, "actual_pm10": None, "actual_pm25": None}


def get_exif_datetime(img):
    """Best-effort EXIF capture-time extraction. Returns None if unavailable."""
    try:
        exif = img.getexif()
        dt_str = None
        if hasattr(exif, "get_ifd"):
            exif_ifd = exif.get_ifd(0x8769)  # Exif SubIFD
            dt_str = exif_ifd.get(0x9003) or exif_ifd.get(0x9004)  # DateTimeOriginal / DateTimeDigitized
        if not dt_str:
            dt_str = exif.get(0x0132)  # DateTime (top-level IFD)
        if not dt_str:
            return None
        return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def estimate_is_night(img):
    """Cheap day/night fallback: mean luminance of a downsized grayscale copy."""
    small = img.convert("L").resize((64, 64))
    return float(np.array(small).mean()) < NIGHT_BRIGHTNESS_THRESHOLD


def select_model_for_image(img, filename):
    """
    Choose Day vs Night model. Tries, in order:
      1) capture time embedded in the internal test-dataset filename
      2) EXIF capture time read from the image
      3) mean-brightness heuristic
    Returns (model, model_name, method, capture_time_used_or_None) so the
    caller can be transparent in the API response about how the decision
    was made — this was previously silent and, for real uploads, always
    fell through to the Day model.
    """
    gt = extract_ground_truth(filename)
    if gt["capture_time"] is not None:
        hour = gt["capture_time"].hour
        is_day = DAY_START_HOUR <= hour < NIGHT_START_HOUR
        model, name = (DAY_MODEL, DAY_MODEL_NAME) if is_day else (NIGHT_MODEL, NIGHT_MODEL_NAME)
        return model, name, "filename_timestamp", gt["capture_time"]

    exif_dt = get_exif_datetime(img)
    if exif_dt is not None:
        hour = exif_dt.hour
        is_day = DAY_START_HOUR <= hour < NIGHT_START_HOUR
        model, name = (DAY_MODEL, DAY_MODEL_NAME) if is_day else (NIGHT_MODEL, NIGHT_MODEL_NAME)
        return model, name, "exif_timestamp", exif_dt

    if estimate_is_night(img):
        return NIGHT_MODEL, NIGHT_MODEL_NAME, "brightness_heuristic", None
    return DAY_MODEL, DAY_MODEL_NAME, "brightness_heuristic", None


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def serve_frontend():
    return send_from_directory(BASE_DIR, "index.html")


@app.errorhandler(404)
def not_found(err):
    if request.path.startswith("/api"):
        return jsonify({"error": "Not found", "path": request.path}), 404
    return "Not Found", 404


@app.errorhandler(413)
def too_large(err):
    return jsonify({"error": f"Upload exceeds the {MAX_UPLOAD_MB} MB limit"}), 413


@app.errorhandler(Exception)
def handle_unexpected_exception(e):
    # Let Flask/Werkzeug handle expected HTTP errors (404, 413, etc.) normally.
    if isinstance(e, HTTPException):
        return e
    logger.exception("Unhandled exception")
    return jsonify({"error": "Internal server error"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "model_loaded_at": MODEL_LOADED_AT,
        "day_model": DAY_MODEL_NAME,
        "night_model": NIGHT_MODEL_NAME,
        "model_error": MODEL_ERROR,
        "db": os.path.exists(DB_PATH),
    })


@app.route("/api/predict", methods=["POST"])
@rate_limited("20 per minute")
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Check /api/health for details."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    filename = file.filename or "upload.jpg"

    # Validate this is really an image before doing anything else with it.
    try:
        file.stream.seek(0)
        probe = Image.open(file.stream)
        probe.verify()
        file.stream.seek(0)
        img = Image.open(file.stream).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        return jsonify({"error": "Uploaded file is not a valid image"}), 400

    latitude = parse_coords(request.form.get("latitude"), -90, 90)
    longitude = parse_coords(request.form.get("longitude"), -180, 180)

    try:
        img_resized = img.resize(INFERENCE_SIZE)
        arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        model, model_name, selection_method, capture_time = select_model_for_image(img, filename)
        logger.info("Prediction using %s (selected via %s)", model_name, selection_method)

        pred = model.predict(arr, verbose=0)[0]
        pm25 = max(0.0, float(pred[0]) * 850)
        pm10 = max(0.0, float(pred[1]) * 1000)

        # Ground-truth comparison (only meaningful for internal test-set filenames)
        gt = extract_ground_truth(filename)
        actual_pm25, actual_pm10 = gt["actual_pm25"], gt["actual_pm10"]
        pm25_error = pm10_error = pm25_accuracy = pm10_accuracy = None
        if actual_pm25 is not None:
            pm25_error = round(pm25 - actual_pm25, 2)
            if actual_pm25 > 0:
                pm25_accuracy = round(max(0, 100 * (1 - abs(pm25 - actual_pm25) / actual_pm25)), 2)
        if actual_pm10 is not None:
            pm10_error = round(pm10 - actual_pm10, 2)
            if actual_pm10 > 0:
                pm10_accuracy = round(max(0, 100 * (1 - abs(pm10 - actual_pm10) / actual_pm10)), 2)

        aqi25, aqi10, aqi_overall, category, color = compute_aqi(pm25, pm10)
        hotspot = is_hotspot_prediction(category, latitude, longitude)

        thumb = img.copy()
        thumb.thumbnail((256, 256))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=75)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        ts = datetime.utcnow().isoformat()

        with get_db() as conn:
            cur = conn.execute("""
                INSERT INTO predictions
                    (timestamp, filename, model_used, pm25, pm10,
                     aqi_pm25, aqi_pm10, aqi_overall, aqi_category, image_b64,
                     latitude, longitude, is_hotspot)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (ts, filename, model_name, pm25, pm10, aqi25, aqi10, aqi_overall,
                  category, img_b64, latitude, longitude, 1 if hotspot else 0))
            record_id = cur.lastrowid

        return jsonify({
            "id": record_id,
            "timestamp": ts,
            "filename": filename,
            "capture_time": capture_time.isoformat() if capture_time else None,
            "model_used": model_name,
            "model_selection_method": selection_method,
            "actual_pm25": actual_pm25,
            "actual_pm10": actual_pm10,
            "pm25_error": pm25_error,
            "pm10_error": pm10_error,
            "pm25_accuracy": pm25_accuracy,
            "pm10_accuracy": pm10_accuracy,
            "pm25": round(pm25, 2),
            "pm10": round(pm10, 2),
            "aqi_pm25": aqi25,
            "aqi_pm10": aqi10,
            "aqi_overall": aqi_overall,
            "aqi_category": category,
            "aqi_color": color,
            "image_b64": img_b64,
            "latitude": latitude,
            "longitude": longitude,
            "is_hotspot": hotspot,
            "geotagged": latitude is not None and longitude is not None,
        })

    except Exception:
        logger.exception("Prediction failed for filename=%s", filename)
        return jsonify({"error": "Prediction failed. Please try a different image."}), 500


@app.route("/api/records", methods=["GET"])
def get_records():
    try:
        limit = min(max(int(request.args.get("limit", 100)), 1), 500)
        offset = max(int(request.args.get("offset", 0)), 0)
    except ValueError:
        limit, offset = 100, 0

    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, timestamp, filename, model_used, pm25, pm10,
                   aqi_pm25, aqi_pm10, aqi_overall, aqi_category,
                   latitude, longitude, is_hotspot
            FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()

    result = [dict(r) for r in rows]
    for row in result:
        row["is_hotspot"] = bool(row.get("is_hotspot"))
        row["geotagged"] = row.get("latitude") is not None and row.get("longitude") is not None
    return jsonify(result)


@app.route("/api/hotspots", methods=["GET"])
def get_hotspots():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, timestamp, filename, pm25, pm10,
                   aqi_pm25, aqi_pm10, aqi_overall, aqi_category,
                   latitude, longitude, is_hotspot
            FROM predictions
            WHERE is_hotspot = 1 AND latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY id DESC
        """).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/geo_records", methods=["GET"])
def get_geo_records():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, timestamp, filename, pm25, pm10,
                   aqi_pm25, aqi_pm10, aqi_overall, aqi_category,
                   latitude, longitude, is_hotspot
            FROM predictions
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY id DESC
        """).fetchall()
    result = [dict(r) for r in rows]
    for row in result:
        row["is_hotspot"] = bool(row.get("is_hotspot"))
    return jsonify(result)


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "google_maps_api_key": "AIzaSyA97xwC2Gu5IjHVwpiX8-JlKzL7Sqn-88I",
        "hotspot_categories": sorted(HOTSPOT_CATEGORIES),
        "default_map_center": DEFAULT_MAP_CENTER,
    })


@app.route("/api/records/<int:record_id>", methods=["GET"])
def get_record(record_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id=?", (record_id,)).fetchone()
    if row is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(dict(row))


@app.route("/api/records/<int:record_id>", methods=["DELETE"])
@require_api_key
def delete_record(record_id):
    with get_db() as conn:
        cur = conn.execute("DELETE FROM predictions WHERE id=?", (record_id,))
        deleted = cur.rowcount
    if not deleted:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"deleted": record_id})


@app.route("/api/stats", methods=["GET"])
def stats():
    with get_db() as conn:
        row = conn.execute("""
            SELECT
              COUNT(*) as total,
              AVG(pm25) as avg_pm25, MAX(pm25) as max_pm25, MIN(pm25) as min_pm25,
              AVG(pm10) as avg_pm10, MAX(pm10) as max_pm10, MIN(pm10) as min_pm10,
              AVG(aqi_overall) as avg_aqi, MAX(aqi_overall) as max_aqi
            FROM predictions
        """).fetchone()
        cat_rows = conn.execute(
            "SELECT aqi_category, COUNT(*) as cnt FROM predictions GROUP BY aqi_category"
        ).fetchall()

    if row["total"] == 0:
        return jsonify({"total": 0})

    return jsonify({
        "total": row["total"],
        "pm25": {"avg": round(row["avg_pm25"], 2), "max": round(row["max_pm25"], 2), "min": round(row["min_pm25"], 2)},
        "pm10": {"avg": round(row["avg_pm10"], 2), "max": round(row["max_pm10"], 2), "min": round(row["min_pm10"], 2)},
        "aqi": {"avg": round(row["avg_aqi"], 1), "max": row["max_aqi"]},
        "category_distribution": {r["aqi_category"]: r["cnt"] for r in cat_rows},
    })


@app.route("/api/aqi_standards", methods=["GET"])
def aqi_standards():
    return jsonify({
        "source": "CPCB – Central Pollution Control Board, India",
        "pm25_breakpoints": [
            {"range": "0–30", "aqi": "0–50", "category": "Good", "health": "Minimal impact"},
            {"range": "30–60", "aqi": "51–100", "category": "Satisfactory", "health": "Minor breathing discomfort to sensitive people"},
            {"range": "60–90", "aqi": "101–200", "category": "Moderate", "health": "Breathing discomfort to asthma patients & elderly"},
            {"range": "90–120", "aqi": "201–300", "category": "Poor", "health": "Breathing discomfort to most on prolonged exposure"},
            {"range": "120–250", "aqi": "301–400", "category": "Very Poor", "health": "Respiratory illness on prolonged exposure"},
            {"range": "250+", "aqi": "401–500", "category": "Severe", "health": "Affects healthy people; serious impact on sensitive groups"},
        ],
        "pm10_breakpoints": [
            {"range": "0–50", "aqi": "0–50", "category": "Good"},
            {"range": "50–100", "aqi": "51–100", "category": "Satisfactory"},
            {"range": "100–250", "aqi": "101–200", "category": "Moderate"},
            {"range": "250–350", "aqi": "201–300", "category": "Poor"},
            {"range": "350–430", "aqi": "301–400", "category": "Very Poor"},
            {"range": "430+", "aqi": "401–500", "category": "Severe"},
        ],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5051)), debug=False)
