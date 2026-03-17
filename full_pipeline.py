# import cv2
# import torch
# import numpy as np
# import os
# import joblib
# import segmentation_models_pytorch as smp

# # Load SVM
# #svm = joblib.load("svm_leaf_model.pkl")

# pca = joblib.load("pca_model.pkl")

# # Load segmentation model
# device = "cpu"

# model = smp.DeepLabV3(
#     encoder_name="resnet50",
#     encoder_weights=None,
#     classes=1,
#     activation=None
# )

# model.load_state_dict(
#     torch.load("deeplabv3_cvppp.pth",
#                map_location=device)
# )

# model.eval()

# # Test image
# img_path = "/Users/sampathvegi/Documents/Python language/python/Projects/leaf-segmentation-classification/CVPPP_1/Natural/A1/plant003_rgb.png"


# img = cv2.imread(img_path)
# img_resized = cv2.resize(img, (256,256))
# img_norm = img_resized / 255.0

# tensor = torch.tensor(img_norm).permute(2,0,1)\
#          .unsqueeze(0).float()

# # Segmentation prediction
# with torch.no_grad():
#     pred = model(tensor)
#     pred = torch.sigmoid(pred)
#     pred = pred.squeeze().numpy()

# mask = (pred > 0.5).astype("uint8") * 255

# # Extract leaves
# contours, _ = cv2.findContours(
#     mask, cv2.RETR_EXTERNAL,
#     cv2.CHAIN_APPROX_SIMPLE
# )

# leaf_id = 0

# for cnt in contours:

#     x,y,w,h = cv2.boundingRect(cnt)

#     if w*h < 200:
#         continue

#     leaf = img_resized[y:y+h, x:x+w]
#     leaf = cv2.resize(leaf, (128,128))

#     # Feature extraction
#     hist = cv2.calcHist(
#         [leaf],[0,1,2],
#         None,[8,8,8],
#         [0,256]*3
#     ).flatten()

#     pred_class = svm.predict([hist])

#     print(f"Leaf {leaf_id} → Class {pred_class[0]}")

#     leaf_id += 1

# print("Pipeline complete ✅")

import cv2
import torch
import numpy as np
import os
import glob
import joblib
import segmentation_models_pytorch as smp


# ==============================
# 1️⃣ Load PCA + SVM Models
# ==============================

pca = joblib.load("pca_model.pkl")
svm = joblib.load("svm_leaf_model.pkl")

print("PCA + SVM models loaded ✅")


# ==============================
# 2️⃣ Load Segmentation Model
# ==============================

device = "cpu"

model = smp.DeepLabV3(
    encoder_name="resnet50",
    encoder_weights=None,
    classes=1,
    activation=None
)

model.load_state_dict(
    torch.load(
        "deeplabv3_cvppp.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("Segmentation model loaded ✅")


# ==============================
# 3️⃣ Select Test Image Automatically
# ==============================

img_list = glob.glob(
    "CVPPP_1/Natural/A1/*_rgb.*"
)

img_path = img_list[0]

print("Using image:", img_path)


# ==============================
# 4️⃣ Read & Prepare Image
# ==============================

img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(
        f"Image not found → {img_path}"
    )

img_resized = cv2.resize(img, (256, 256))
img_norm = img_resized / 255.0

tensor = torch.tensor(img_norm)\
    .permute(2, 0, 1)\
    .unsqueeze(0)\
    .float()


# ==============================
# 5️⃣ Segmentation Prediction
# ==============================

with torch.no_grad():
    pred = model(tensor)
    pred = torch.sigmoid(pred)
    pred = pred.squeeze().numpy()

mask = (pred > 0.5).astype("uint8") * 255


# ==============================
# 6️⃣ Extract Individual Leaves
# ==============================

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

leaf_id = 0

for cnt in contours:

    x, y, w, h = cv2.boundingRect(cnt)

    if w * h < 200:
        continue

    leaf = img_resized[y:y+h, x:x+w]
    leaf = cv2.resize(leaf, (128, 128))


    # ==============================
    # 7️⃣ Feature Extraction
    # ==============================

    gray = cv2.cvtColor(leaf, cv2.COLOR_BGR2GRAY)

    # Background removal
    _, thresh = cv2.threshold(
        gray, 240, 255,
        cv2.THRESH_BINARY_INV
    )

    leaf_masked = cv2.bitwise_and(
        leaf, leaf,
        mask=thresh
    )

    # Color histogram
    hist = cv2.calcHist(
        [leaf_masked],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256] * 3
    ).flatten()

    # Hu moments
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten()

    # Texture
    texture = np.std(gray)

    feature_vector = np.hstack(
        [hist, hu, texture]
    )


    # ==============================
    # 8️⃣ Apply PCA Transform
    # ==============================

    feature_vector = pca.transform(
        [feature_vector]
    )


    # ==============================
    # 9️⃣ Predict Leaf Class
    # ==============================

    pred_class = svm.predict(
        feature_vector
    )

    print(
        f"Leaf {leaf_id} → Class {pred_class[0]}"
    )

    leaf_id += 1


print("\nPipeline complete ✅")
