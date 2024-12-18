import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import sqrtm
from PIL import Image
import os
from torch.utils.data import Dataset



class ImageDataset(Dataset):
    def __init__(self, folder_path1, folder_path2, transform=None):
        self.folder_path1 = folder_path1
        self.folder_path2 = folder_path2
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path1) if os.path.isfile(os.path.join(folder_path1, f)) and (f.endswith(".jpg") or f.endswith(".png"))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path1 = os.path.join(self.folder_path1, self.image_files[idx])
        image1 = Image.open(img_path1).convert("RGB")  # Ensures 3-channel images

        img_path2 = os.path.join(self.folder_path2, self.image_files[idx])
        image2 = Image.open(img_path2).convert("RGB")  # Ensures 3-channel images

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2



def calculate_fid(path1, path2, batch_size=32, device="cuda", method="fid"):
    # Define the transform for pre-processing images
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 expects 299x299
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] range
    ])

    # Load datasets
    dataset = ImageDataset(path1, path2, transform=transform)
    
    # Create data loaders
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load InceptionV3 model
    inception = models.inception_v3(pretrained=True)
    inception.fc = nn.Identity()  # Remove fully connected layer for feature extraction
    inception.eval().to(device)

    def get_activations(loader):
        """Extract InceptionV3 features for a given dataset loader."""
        activations1 = []
        activations2 = []
        with torch.no_grad():
            for images in loader:
                img1, img2 = images
                img1, img2 = img1.to(device), img2.to(device)
                features1 = inception(img1)
                features2 = inception(img2)
                activations1.append(features1.cpu().numpy())
                activations2.append(features2.cpu().numpy())
        activations1 = np.concatenate(activations1, axis=0)
        activations2 = np.concatenate(activations2, axis=0)
        return activations1, activations2

    # Calculate activations for both datasets
    act1, act2 = get_activations(loader)

    if method == "cosine":
        return (torch.nn.functional.normalize(torch.tensor(act1), dim=-1)*torch.nn.functional.normalize(torch.tensor(act2), dim=-1)).sum(-1).mean()

    elif method == "fid":
        # Compute means and covariances
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        # Compute FID score
        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)  # Matrix square root of the product of covariances
        if np.iscomplexobj(covmean):
            covmean = covmean.real  # Correct potential imaginary component

        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid
    
    raise ValueError(f"Unknown method: {method}")



# Example usage
# fid_value = calculate_fid("outputs_simple/KV_Merge_25Percent_Mean_OnKey_2layerskip", "outputs_simple/BaseModel")
fid_value = calculate_fid("outputs_simple/Q_Merge_25Percent_Mean_25layerskip", "outputs_simple/BaseModel")
print(f"FID: {fid_value}")




# What does the ODE trajectory look like for each token?
# Simpler dataset with a single subject to see how it changes
# distances over # toks merged
# distances when adding token merging to more and more layers (starting from most similar)
# merging before and after splitting into heads (head redundancy?)
# Annotate paper with dimension
# Sum vs mean