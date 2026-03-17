import cv2
import numpy as np
import os
import torch
import segmentation_models_pytorch as smp

# Device
device = "cpu"

# Load trained model
model = smp.DeepLabV3(
    encoder_name="resnet50",
    encoder_weights=None,
    classes=1,
    activation=None
)

model.load_state_dict(torch.load("deeplabv3_cvppp.pth", map_location=device))
model.to(device)
model.eval()

# Output folder
os.makedirs("segmented_leaves", exist_ok=True)

# Test image
img_path = "CVPPP_1/Natural/A1/plant003_rgb.png"

img = cv2.imread(img_path)
img_resized = cv2.resize(img, (256,256))
img_norm = img_resized / 255.0

tensor = torch.tensor(img_norm).permute(2,0,1).unsqueeze(0).float()


# Predict mask
with torch.no_grad():
    pred = model(tensor)
    pred = torch.sigmoid(pred)
    pred = pred.squeeze().numpy()

# Convert to binary
mask = (pred > 0.5).astype("uint8") * 255

kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Predicted Mask")

plt.show()

# Find contours
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

leaf_id = 0

for cnt in contours:

    x,y,w,h = cv2.boundingRect(cnt)

    if w*h < 200:   # remove tiny noise
        continue
    
    leaf = img_resized[y:y+h, x:x+w]

    cv2.imwrite(
        f"segmented_leaves/leaf_{leaf_id}.png",
        leaf
    )

    leaf_id += 1

print("Leaves extracted:", leaf_id)
