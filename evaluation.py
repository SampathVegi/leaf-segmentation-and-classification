import pandas as pd
import cv2
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==============================
# 1️⃣ Load Models
# ==============================

svm = joblib.load("svm_leaf_model.pkl")
pca = joblib.load("pca_model.pkl")

print("Models loaded ✅")


# ==============================
# 2️⃣ Load Dataset
# ==============================

df = pd.read_csv("Flavia/all.csv")

image_folder = "Flavia/"

images = []
labels = []

for _, row in df.iterrows():

    img_path = os.path.join(
        image_folder,
        row["id"]
    )

    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.resize(img, (128,128))

    images.append(img)
    labels.append(row["y"])

print("Images loaded:", len(images))


# ==============================
# 3️⃣ Feature Extraction
# ==============================

features = []

for img in images:

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray, 240, 255,
        cv2.THRESH_BINARY_INV
    )

    leaf = cv2.bitwise_and(
        img, img,
        mask=thresh
    )

    # Histogram
    hist = cv2.calcHist(
        [leaf],
        [0,1,2],
        None,
        [8,8,8],
        [0,256]*3
    ).flatten()

    # Hu Moments
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(
        moments
    ).flatten()

    # Texture
    texture = np.std(gray)

    feature_vector = np.hstack(
        [hist, hu, texture]
    )

    features.append(feature_vector)

X = np.array(features)
y_true = np.array(labels)

print("Feature shape:", X.shape)


# ==============================
# 4️⃣ PCA Transform
# ==============================

X = pca.transform(X)


# ==============================
# 5️⃣ Predictions
# ==============================

y_pred = svm.predict(X)


# ==============================
# 6️⃣ Compute Metrics
# ==============================

accuracy = accuracy_score(
    y_true, y_pred
)

precision = precision_score(
    y_true, y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true, y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true, y_pred,
    average="weighted",
    zero_division=0
)

print("\nEvaluation Metrics")
print("------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ==============================
# 7️⃣ Metrics Bar Plot
# ==============================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

values = [
    accuracy,
    precision,
    recall,
    f1
]

plt.figure(figsize=(8,5))

plt.bar(metrics, values)

plt.ylim(0,1)

plt.title(
    "Classification Evaluation Metrics"
)

plt.ylabel("Score")

for i, v in enumerate(values):
    plt.text(
        i,
        v + 0.02,
        f"{v:.2f}",
        ha="center"
    )

plt.show()


# ==============================
# 8️⃣ Confusion Matrix
# ==============================

cm = confusion_matrix(
    y_true, y_pred
)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.title("Confusion Matrix")

plt.show()
