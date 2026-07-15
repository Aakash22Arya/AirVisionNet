import os
import numpy as np
from PIL import Image
from keras.models import load_model
import pandas as pd


model = load_model('/mainProject/MainModel_2.h5')

# Path to test folder
test_path = '/a/mainProject/PM_Night_Annoted_train/'  

# Get test files (you can reuse your test_files list too)
test_files = os.listdir(test_path)

# Image size
IMG_SIZE = (1024, 1024)

# Store results
results = []


for f in test_files:
    try:
        # ---- Load and preprocess image ----
        img = Image.open(os.path.join(test_path, f))
        img = img.resize(IMG_SIZE)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # ---- Extract ground truth from filename ----
        f1 = f.split('@')
        z1 = f1[1]
        z2 = f1[2]

        gt1 = float(z1) / 850
        gt2 = float(z2[:-4]) / 1000  # remove .jpg/.png

        gt = np.array([gt1, gt2])

        # ---- Predict ----
        pred = model.predict(img)[0]

        # ---- Denormalize predictions ----
        pred1 = pred[0] * 850
        pred2 = pred[1] * 1000

        gt1_actual = gt1 * 850
        gt2_actual = gt2 * 1000
        # ---- Save result ----
        results.append({
            "filename": f,
            "GT_PM1": gt1_actual,
            "GT_PM2": gt2_actual,
            "Pred_PM1": pred1,
            "Pred_PM2": pred2
        })

        print(f"Processed: {f}")

    

    except Exception as e:
        print(f"Error processing {f}: {e}")

#  Convert to numpy
df = pd.DataFrame(results)
csv_path = "/mainProject/prediction_results.csv"
df.to_csv(csv_path, index=False)

print(f"\nResults saved to {csv_path}")
