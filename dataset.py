import os
import cv2
import torch
from torch.utils.data import Dataset

class CVPPPDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.masks = []

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith("_rgb.png"):

                    img_path = os.path.join(root, file)
                    mask_path = img_path.replace("_rgb.png", "_fg.png")

                    if os.path.exists(mask_path):
                        self.images.append(img_path)
                        self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.resize(img, (256,256))
        img = img / 255.0
        img = torch.tensor(img).permute(2,0,1).float()

        mask = cv2.imread(self.masks[idx], 0)
        mask = cv2.resize(mask, (256,256))
        mask = mask / 255.0
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask
