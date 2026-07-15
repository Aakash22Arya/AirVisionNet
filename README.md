# AirVisionNet – Air Quality Estimation System

AirVisionNet is a CNN-based air quality estimation system that predicts **PM2.5**, **PM10**, and the **Air Quality Index (AQI)** from outdoor RGB images captured during both **daytime** and **nighttime**. The system provides real-time air quality analysis through a Flask-based backend and a web-based frontend.

---

# Features

* Predicts **PM2.5** and **PM10** concentrations from a single outdoor RGB image.
* Calculates **Air Quality Index (AQI)** using CPCB (Central Pollution Control Board, India) standards.
* Supports both **daytime** and **nighttime** image prediction.
* Interactive web interface for image upload and result visualization.
* Stores prediction history in a SQLite database.
* Displays AQI category with corresponding color coding.
* RESTful API for prediction and data retrieval.

---

# Project Structure

```text
AirVisionNet/
├── app.py                     # Flask backend
├── index.html                 # Frontend
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── LightairnetMainModel.h5    # Daytime prediction model
├── MainModel_2.h5             # Nighttime prediction model
├── predictions.db             # SQLite database (created automatically)
└── uploads/                   # Uploaded images (created automatically)
```

---

# System Requirements

## Software

* Python **3.10.x**
* pip
* Modern web browser (Chrome, Edge, Firefox)

## Python Packages

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

# Installation

## 1. Clone or Download the Project

Extract or clone the project folder.

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Verify Model Files

Ensure the following model files are present in the project directory:

* `LightairnetMainModel.h5`
* `MainModel_2.h5`

## 4. Start the Backend Server

```bash
python app.py
```

The server starts at:

```
http://localhost:5051
```

---

# Running the Frontend

## Option 1 (Recommended)

Start a simple HTTP server:

```bash
python -m http.server 8080
```

Open your browser and navigate to:

```
http://localhost:8080
```

## Option 2

Open `index.html` directly in a web browser.

---

# API Endpoints

| Method | Endpoint             | Description                    |
| ------ | -------------------- | ------------------------------ |
| GET    | `/api/health`        | Check backend and model status |
| POST   | `/api/predict`       | Predict PM2.5, PM10 and AQI    |
| GET    | `/api/records`       | Retrieve prediction history    |
| GET    | `/api/records/<id>`  | Retrieve a specific prediction |
| DELETE | `/api/records/<id>`  | Delete a prediction record     |
| GET    | `/api/stats`         | Retrieve summary statistics    |
| GET    | `/api/aqi_standards` | CPCB AQI standards             |

---

# Example Prediction Response

```json
{
  "id": 1,
  "timestamp": "2026-07-14T12:00:00",
  "filename": "sample_image.jpg",
  "pm25": 62.4,
  "pm10": 134.7,
  "aqi_pm25": 103,
  "aqi_pm10": 110,
  "aqi_overall": 110,
  "aqi_category": "Moderate"
}
```

---

# AQI Standard

AirVisionNet calculates AQI according to the **Central Pollution Control Board (CPCB), Government of India** methodology.

The overall AQI is determined as the maximum of the PM2.5 and PM10 sub-indices.

---

# Database

Prediction records are automatically stored in a SQLite database named:

```
predictions.db
```

Stored information includes:

* Prediction ID
* Timestamp
* Uploaded filename
* PM2.5 concentration
* PM10 concentration
* AQI values
* AQI category
* Thumbnail image

---

# Image Processing Pipeline

```
Outdoor RGB Image
        │
        ▼
Image Preprocessing
(Resize & Normalize)
        │
        ▼
CNN Prediction Model
        │
        ▼
PM2.5 & PM10 Prediction
        │
        ▼
AQI Calculation
(CPCB Standard)
        │
        ▼
Web Dashboard Display
```

---

# Technologies Used

* Python 3.10
* TensorFlow
* Keras
* Flask
* Flask-CORS
* NumPy
* Pandas
* Pillow
* SQLite
* HTML
* CSS
* JavaScript

---

# Notes

* Input images are resized to **1024 × 1024** before prediction.
* AirVisionNet supports prediction for both **daytime** and **nighttime** outdoor RGB images.
* AQI values are computed according to **CPCB (India)** standards.
* Prediction history is automatically stored for future reference.

---

# License

This project is developed for academic and research purposes.
