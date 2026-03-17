import torch
import cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else "cpu"
# Load model
model = smp.DeepLabV3(
    encoder_name="resnet50",
    encoder_weights=None,
    classes=1,
    activation=None
)

model.load_state_dict(torch.load("deeplabv3_cvppp.pth"))
model.to(device)
model.eval()

# Load test image
img = cv2.imread("CVPPP/Natural/A1/plant001_rgb.png")
img_resized = cv2.resize(img, (256,256))
img_norm = img_resized / 255.0

tensor = torch.tensor(img_norm).permute(2,0,1).unsqueeze(0).float().to(device)

# Predict
with torch.no_grad():
    pred = model(tensor)
    pred = torch.sigmoid(pred)
    pred = pred.squeeze().cpu().numpy()

# Show output
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(pred, cmap="gray")
plt.title("Segmented Mask")

plt.show()
