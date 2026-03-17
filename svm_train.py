import pandas as pd
import cv2
import os
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import joblib


# ==============================
# 1️⃣ Load CSV Labels
# ==============================

df = pd.read_csv("Flavia/all.csv")

print(df.head())
print("Total records:", len(df))


# ==============================
# 2️⃣ Load Images
# ==============================

image_folder = "Flavia/"

images = []
labels = []

for _, row in df.iterrows():

    img_path = os.path.join(image_folder, row["id"])

    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.resize(img, (128, 128))

    images.append(img)
    labels.append(row["y"])

print("Images loaded:", len(images))


# ==============================
# 3️⃣ Feature Extraction
# ==============================

features = []

for img in images:

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Background removal
    _, thresh = cv2.threshold(
        gray, 240, 255,
        cv2.THRESH_BINARY_INV
    )

    leaf = cv2.bitwise_and(
        img, img,
        mask=thresh
    )

    # ---- Color Histogram ----
    hist = cv2.calcHist(
        [leaf],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256] * 3
    )
    hist = hist.flatten()

    # ---- Shape Features (Hu Moments) ----
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten()

    # ---- Texture Feature ----
    texture = np.std(gray)

    # Combine all features
    feature_vector = np.hstack(
        [hist, hu, texture]
    )

    features.append(feature_vector)


X = np.array(features)
y = np.array(labels)

print("Feature shape before PCA:", X.shape)


# ==============================
# 4️⃣ PCA Dimensionality Reduction
# ==============================

pca = PCA(n_components=100)

X = pca.fit_transform(X)

print("After PCA shape:", X.shape)

# Save PCA model
joblib.dump(pca, "pca_model.pkl")
print("PCA model saved ✅")


# ==============================
# 5️⃣ Train/Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# 6️⃣ Train SVM Classifier
# ==============================

svm = SVC(kernel="rbf")

svm.fit(X_train, y_train)

print("SVM training complete ✅")


# ==============================
# 7️⃣ Evaluate Accuracy
# ==============================

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Classification Accuracy:", accuracy)


# ==============================
# 8️⃣ Save SVM Model
# ==============================

joblib.dump(svm, "svm_leaf_model.pkl")

print("SVM model saved ✅")
