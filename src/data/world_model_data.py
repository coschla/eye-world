import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

from image import CustomImageDataset
import torch
from torchvision.transforms import ToTensor
from torchvision import transforms, datasets
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

import torch
from torch.utils.data import Dataset, DataLoader
from image import CustomImageDataset  # Assuming you have this dataset class defined
import os
from torchvision import transforms
from torchvision.io import read_image

class world_data(Dataset):
    def __init__(self):
        # Initialize the dataset
        raw_image = CustomImageDataset()  # Load the custom dataset
        
        num_samples = len(raw_image)  # Get the number of samples from the dataset
        num_samples = num_samples - 200  # Adjust if needed

        self.data = torch.zeros(num_samples, 3, 210, 210)  # Three channels, 210x210 size for each image
        for i in range(num_samples):
            # Access images from the dataset; only image returned (no tuple)
            img0 = raw_image[i]  # First image (noisy image)
            img1 = raw_image[i + 100]  # Second image (noisy image)
            img2 = raw_image[i + 2 * 100]  # Third image (clean image)

            # Store the images into the data tensor
            self.data[i, 0] = img0.squeeze(0)  # First input image
            self.data[i, 1] = img1.squeeze(0)  # Second input image
            self.data[i, 2] = img2.squeeze(0)  # Target image

    def __len__(self):
        return len(self.data)  # Number of samples (images)

    def __getitem__(self, idx):
        # Return two input images and the target image (three images in total)
        img1 = self.data[idx, 0]  # First image
        img2 = self.data[idx, 1]  # Second image
        img3 = self.data[idx, 2]  # Target image
        return img1, img2, img3
