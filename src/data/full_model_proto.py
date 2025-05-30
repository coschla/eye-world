
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
#from world_model import world_data
import torch
from torchvision.transforms import ToTensor
from torchvision import transforms, datasets
import os
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import re
from image import CustomImageDataset
from eye_gaze_pred import CustomTransformer

class WorldData(Dataset):
    def __init__(self):
        raw_image = CustomImageDataset(image_dir= r"C:\Users\X570 MASTER\Desktop\data\game_frames\Class1")  # Specify the path to the images
        
        num_samples = len(raw_image)
        num_samples = num_samples - 200  # Adjust if needed

        self.data = torch.zeros(num_samples, 3, 210, 210)  # Three channels, 210x210 size for each image
        for i in range(num_samples):
            img0 = raw_image[i]  # First image (noisy image)
            img1 = raw_image[i + 100]  # Second image (noisy image)
            img2 = raw_image[i + 2 * 100]  # Third image (clean image)

            self.data[i, 0] = img0.squeeze(0)  # First input image
            self.data[i, 1] = img1.squeeze(0)  # Second input image
            self.data[i, 2] = img2.squeeze(0)  # Target image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1 = self.data[idx, 0]  # First image
        img2 = self.data[idx, 1]  # Second image
        img3 = self.data[idx, 2]  # Target image
        return img1, img2, img3








class ZoomBox166(nn.Module):
    """
    Crop a square patch from the image and upscale it to 166x166,
    matching the downsample effect of two 25x25 convs (210→188→166).
    """
    def __init__(self, box_size: int):
        super().__init__()
        self.box_size = box_size
        self.out_size = 166  # target spatial dimension
        gaze_model = CustomTransformer()

        # Load weights
        gaze_model.load_state_dict(torch.load(r"C:\Users\X570 MASTER\Desktop\data\custom_conv_500.pth"))
        gaze_model.eval()
        gaze_model.to(device) 

    def forward(self, x: torch.Tensor, coords: tuple[int, int]) -> torch.Tensor:
        # x: [B, C, 210, 210], coords: (x_center, y_center)
        B, C, H, W = x.shape

        x0, y0 = coords

        half = self.box_size // 2
        left   = max(0, x0 - half)
        right  = min(W, x0 + half)
        top    = max(0, y0 - half)
        bottom = min(H, y0 + half)

        patch = x[:, :, top:bottom, left:right]
        zoomed = F.interpolate(patch,
                               size=(self.out_size, self.out_size),
                               mode='bilinear',
                               align_corners=False)
        return zoomed











class ImageGenerator(nn.Module):
    def __init__(self, box_size: int = 83):
        super(ImageGenerator, self).__init__()

        # Replace first two conv layers with ZoomBox166
        self.zoom = ZoomBox166(box_size)

        # Continue with remaining convs (conv3…conv8)
        self.conv3_img1 = nn.Conv2d(1,   48, kernel_size=3, stride=1, padding=1)
        self.conv4_img1 = nn.Conv2d(48,  64, kernel_size=3, stride=1, padding=1)
        self.conv5_img1 = nn.Conv2d(64,  80, kernel_size=3, stride=1, padding=1)
        self.conv6_img1 = nn.Conv2d(80,  96, kernel_size=3, stride=1, padding=1)
        self.conv7_img1 = nn.Conv2d(96, 112, kernel_size=3, stride=1, padding=1)
        self.conv8_img1 = nn.Conv2d(112,128, kernel_size=3, stride=1, padding=1)

        self.drop = nn.Dropout(0.1)
        self.relu = nn.LeakyReLU()

        # Decoder: mirror conv3…conv8, then invert the ZoomBox effect
        self.deconv1_img1 = nn.ConvTranspose2d(128,112, kernel_size=3, stride=1, padding=1)
        self.deconv2_img1 = nn.ConvTranspose2d(112, 96, kernel_size=3, stride=1, padding=1)
        self.deconv3_img1 = nn.ConvTranspose2d(96,  80, kernel_size=3, stride=1, padding=1)
        self.deconv4_img1 = nn.ConvTranspose2d(80,  64, kernel_size=3, stride=1, padding=1)
        self.deconv5_img1 = nn.ConvTranspose2d(64,  48, kernel_size=3, stride=1, padding=1)
        self.deconv6_img1 = nn.ConvTranspose2d(48,  32, kernel_size=3, stride=1, padding=1)
        # invert two 25x25 convs: upsample back 166→188→210
        self.deconv7_img1 = nn.ConvTranspose2d(32,  16, kernel_size=25, stride=1, padding=1)
        self.deconv8_img1 = nn.ConvTranspose2d(16,   1,  kernel_size=25, stride=1, padding=1)


        self.gaze_model = CustomTransformer()

        # Step 2: Load weights with check
        path = r"C:\Users\X570 MASTER\Desktop\data\custom_conv_500.pth"
        try:
            state_dict = torch.load(path, map_location='cpu')  # map to CPU first to avoid CUDA mismatch
            self.gaze_model.load_state_dict(state_dict)
            print(f"[✔] Loaded pretrained gaze model from {path}")
        except Exception as e:
            print(f"[✖] Failed to load gaze model from {path}: {e}")

        self.gaze_model.eval()
        for param in self.gaze_model.parameters():
            param.requires_grad = False


    def forward(self, img_1: torch.Tensor, img_2: torch.Tensor, coords: tuple[int,int]) -> torch.Tensor:
        
        gaze_coords = self.gaze_model(img_2)  # [B, 2]
        x0, y0 = gaze_coords[0]  # just use the first image if batch size is 1
        x0 = int(x0.item())
        y0 = int(y0.item())

        x = self.zoom(img_2, coords=(x0, y0)) 
        x = self.relu(self.conv3_img1(x))      # 1→48
        x = self.drop(x)
        x = self.relu(self.conv4_img1(x))      # 48→64
        x = self.relu(self.conv5_img1(x))      # 64→80
        x = self.relu(self.conv6_img1(x))      # 80→96
        x = self.drop(x)
        x = self.relu(self.conv7_img1(x))      # 96→112
        x = self.relu(self.conv8_img1(x))      # 112→128

        x = self.relu(self.deconv1_img1(x))    # 128→112
        x = self.relu(self.deconv2_img1(x))    # 112→96
        x = self.drop(x)
        x = self.relu(self.deconv3_img1(x))    # 96→80
        x = self.relu(self.deconv4_img1(x))    # 80→64
        x = self.relu(self.deconv5_img1(x))    # 64→48
        x = self.drop(x)
        x = self.relu(self.deconv6_img1(x))    # 48→32
        x = self.relu(self.deconv7_img1(x))    # 32→16 (→188×188)
        x = self.deconv8_img1(x)               # 16→1  (→210×210)
        return x


# Define the Dataset Class (world_data)
class WorldData(Dataset):
    def __init__(self):
        raw_image = CustomImageDataset()  # Load the custom dataset
        num_samples = len(raw_image)
        print(num_samples)
        num_samples = 17000  # Adjust if needed

        self.data = torch.zeros(num_samples, 3, 210, 210)  # Three channels, 210x210 size for each image
        for i in range(num_samples):
            img0 = raw_image[i]  # First image (noisy image)
            img1 = raw_image[i + 100]  # Second image (noisy image)
            img2 = raw_image[i + 2 * 100]  # Third image (clean image)

            self.data[i, 0] = img0.squeeze(0)  # First input image
            self.data[i, 1] = img1.squeeze(0)  # Second input image
            self.data[i, 2] = img2.squeeze(0)  # Target image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1 = self.data[idx, 0]  # First image
        img2 = self.data[idx, 1]  # Second image
        img3 = self.data[idx, 2]  # Target image
        return img1, img2, img3

# Eye Gaze Model Training Loop
class EyeGazeImageGenerator:
    def __init__(self):
        print('Initializing EyeGazeImageGenerator...')
        # Initialize the model, optimizer, and loss function
        self.model = ImageGenerator().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4,weight_decay=1e-2)
        self.criterion = nn.L1Loss()


        '''
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',  # minimize validation loss
            factor=0.5,  # multiply lr by 0.5 when triggered
            patience=3,  # wait 3 epochs with no improvement
            verbose=True  # print a message when lr is reduced
        )'''

        # Load data
        full_data = WorldData()  # Assuming WorldData is already defined
        train_size = int(0.9 * len(full_data))  # 90% for training
        val_size = len(full_data) - train_size  # 10% for validation

        # Split data into training and validation sets
        self.train_data, self.val_data = random_split(full_data, [train_size, val_size])

        self.train_loader = DataLoader(self.train_data, batch_size=1, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=1, shuffle=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train(self, num_epochs=10):
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            total_loss_train = 0
            for step, (img_1, img_2, img_3) in enumerate(self.train_loader):
                img_1, img_2, img_3 = img_1.to(self.device), img_2.to(self.device), img_3.to(self.device)

                # Forward pass
                output = self.model(img_1, img_2)

                # Compute loss
                loss = self.criterion(output, img_3)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss_train += loss.item()

            # Average loss for training
            avg_loss_train = total_loss_train / len(self.train_loader)
            train_losses.append(avg_loss_train)

            # Validation phase
            self.model.eval()
            total_loss_val = 0
            with torch.no_grad():  # No need to compute gradients for validation
                for step, (img_1, img_2, img_3) in enumerate(self.val_loader):
                    img_1, img_2, img_3 = img_1.to(self.device), img_2.to(self.device), img_3.to(self.device)

                    # Forward pass
                    output = self.model(img_1, img_2)

                    # Compute loss
                    loss = self.criterion(output, img_3)

                    total_loss_val += loss.item()

            # Average loss for validation
            avg_loss_val = total_loss_val / len(self.val_loader)
            val_losses.append(avg_loss_val)

            # Print the average loss for the epoch
            print(f"Epoch {epoch+1}, Train Loss: {avg_loss_train:.4f}, Val Loss: {avg_loss_val:.4f}")

            # Adjust the learning rate based on validation loss
            #self.scheduler.step(avg_loss_val)

            # Save model checkpoint
            if avg_loss_val < 0.1:
                torch.save(self.model.state_dict(), f"model_epoch_{epoch+1}.pth")

        # After training, plot the loss graphs
        self.plot_loss_graph(train_losses, val_losses)

    def plot_loss_graph(self, train_losses, val_losses):
        epochs = np.arange(1, len(train_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label="Train Loss", color='blue')
        plt.plot(epochs, val_losses, label="Validation Loss", color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.legend()
        plt.show()

# Example usage:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eye_gaze = EyeGazeImageGenerator()
eye_gaze.train(num_epochs=25)
