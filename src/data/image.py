"""

import torch
from torchvision.transforms import ToTensor
from torchvision import transforms, datasets
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import re


def __len__(self):
    return len(self.img_labels)


transform = transforms.Compose([
    transforms.Pad((25, 0, 25, 0)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

class CustomImageDataset(Dataset):
    def __init__(self, transform=transform):
        self.transform = transform
        self.root = r"src\data\game_frames"


        self.image_paths = sorted(
            [
                os.path.join(self.root, fname)
                for fname in os.listdir(self.root)
                if fname.endswith(('.png', '.jpg', '.jpeg'))
            ],
            key=lambda x: int(re.findall(r"_(\d+)", x)[-1])  # grabs the last number after an underscore
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image
"""


# image name RZ_5037271_39

import torch
from torchvision.transforms import ToTensor
from torchvision import transforms, datasets
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import re


class CustomImageDataset(Dataset):
    def __init__(self, transform=None):
        print("Initializing CustomImageDataset...")
        self.to_pil = (
            transforms.ToPILImage()
        )  # Converts tensor to PIL Image for transformation
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Pad((0, 0, 0, 0)),
                transforms.ToTensor(),
            ]
        )

        # Use provided transform if given
        if transform is not None:
            self.transform = transform
        self.root = r"C:\Users\X570 MASTER\Desktop\data\game_frames\Class1"

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Directory not found: {self.root}")

        # Gather all image paths
        self.image_paths = sorted(
            [
                os.path.join(self.root, fname)
                for fname in os.listdir(self.root)
                if fname.lower().endswith((".png", ".jpg", ".jpeg"))
            ],
            key=lambda x: int(re.findall(r"_(\d+)", x)[-1])
            if re.findall(r"_(\d+)", x)
            else 0,
        )

        #print(f"  ➜ Found {len(self.image_paths)} images in '{self.root}'")

    def __len__(self):
        length = len(self.image_paths)
        #print(f"__len__ called, returning {length}")
        return length

    def __getitem__(self, idx):
        #print(f"__getitem__ called with idx={idx}")
        img_path = self.image_paths[idx]
        #print(f"  ➜ Loading image from: {img_path}")

        image = read_image(img_path)
        image = self.transform(image)
        return image
        #print(f"  ➜ Raw image tensor shape: {image.shape}, dtype={image.dtype}")

        #if self.transform:
            #image = self.transform(image)
            #print(f"  ➜ After transform: shape={image.shape}, dtype={image.dtype}")
        #else:
            #print("  ➜ No transform applied.")

        #return image


# Quick sanity check
if __name__ == "__main__":
    dataset = CustomImageDataset()
    print("Dataset length:", len(dataset))
    sample = dataset[0]
    print("Sample tensor:", sample)

    """
    def __init__(self, transform=transform):
        print("Initializing CustomImageDataset...")
        self.transform = transform
        self.root = r"src\data\game_frames\Class1"
        
        # gather all image paths
        self.image_paths = sorted(
            [
                os.path.join(self.root, fname)
                for fname in os.listdir(self.root)
                if fname.endswith(('.png', '.jpg', '.jpeg'))
            ],
            key=lambda x: int(re.findall(r"_(\d+)", x)[-1])  # grabs the last number after an underscore
        )
        print(f"  ➜ Found {len(self.image_paths)} images in '{self.root}'")
"""
