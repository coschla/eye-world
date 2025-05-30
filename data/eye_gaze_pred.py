'''
import torch
import torch.nn as nn
import torch.optim as optim


# import math
import numpy as np

from eye_data_sorting import EyeData

# import csv
import matplotlib.pyplot as plt

print("a")

"""                                
class CustomTransformer(nn.Module):
    def __init__(self):                                               
        super().__init__()                                                                                                               

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(210 * 210, 3000)
        self.layer2 = nn.Linear(3000, 2000)
        self.layer3 = nn.Linear(2000, 1000)
        self.layer4 = nn.Linear(1000, 500)
        self.layer5 = nn.Linear(500, 100)
        self.layer6 = nn.Linear(100, 2)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # x could be [H,W], [C,H,W], or [B,C,H,W]

        # 1) Batch‐dim
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # → [1,1,H,W]
        elif x.dim() == 3:
            x = x.unsqueeze(0)  # → [1,C,H,W]

        # 3) ViTModel expects pixel_values=…
        x = self.flatten(x)
        outputs = self.layer1(x)
        outputs = self.relu(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.relu(outputs)
        outputs = self.layer4(outputs)
        outputs = self.layer5(outputs)
        outputs = self.relu(outputs)
        outputs = self.layer6(outputs)

        coords = self.sigmoid(outputs) * 210.0
        return coords
"""


class CustomTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
        nn.Conv2d(1,16,kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.3),
        nn.MaxPool2d(2),


        nn.Conv2d(16,32,kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.3),
        nn.MaxPool2d(2),
 

        nn.Conv2d(32,64,kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.3),
        nn.MaxPool2d(2),


        )

        self.flat=nn.Flatten()

        self.fin=nn.Sequential(
            nn.Linear(64*20*26,256),
            nn.LeakyReLU(),
            nn.Dropout(.4),
            nn.Linear(256,2),

        )
       # self.register_buffer('scale', torch.tensor([160.0, 210.0]))

        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        # x could be [H,W], [C,H,W], or [B,C,H,W]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) Batch‐dim

        if x.dim() == 2:           # [H,W]
            x = x.unsqueeze(0).unsqueeze(0)  # → [1,1,H,W]
        elif x.dim() == 3:         # [C,H,W]
            x = x.unsqueeze(0)              # → [1,C,H,W]
        


        #coords=self.sigmoid(coords)
        feats = self.conv(x)
        feats = self.flat(feats)

        coords = self.fin(feats)                 # → [B,2]
        #norm = (torch.tanh(raw) + 1.0) * 0.5   # → [B,2] in [0,1]
        #coords = norm * self.scale  

        return coords
     





class eye_gaze_(nn.Module):
    def __init__(self):
        model = CustomTransformer()

        criterion = nn.MSELoss()

        # Optimizer

        optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)
        #criterion = nn.SmoothL1Loss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',       # we want to minimize validation loss
            factor=0.5,       # multiply lr by 0.5 when triggered
            patience=3,       # wait 3 epochs with no improvement
            verbose=True      # print a message when lr is reduced
            )  


        eye_data_instance = EyeData()

        # Initialize lists to store data
        data_list = []

        # Loop through the entire dataset to collect all images and coordinates
        for idx in range(len(eye_data_instance)):
            sample_data = eye_data_instance[idx]  # Get the combined data tensor

            if sample_data is None:  # Skip samples with missing gaze data
                continue

            # Append the sample data tensor (which includes both image and coordinates)
            data_list.append(
                sample_data.unsqueeze(0)
            )  # Add an extra batch dimension for consistency

        # Stack all data into a single tensor (each entry is a combined tensor of image + coordinates)
        all_data = torch.cat(data_list, dim=0)

        # Now, split the data into training and testing groups (85% training, 15% testing)
        train_size = int(0.85 * len(all_data))
        test_size = len(all_data) - (train_size)

        # Shuffle the dataset to randomize the split
        indices = np.random.permutation(len(all_data))

        train_indices, test_indices = indices[:train_size], indices[train_size:]

        # Create the training and testing datasets
        train_data = all_data[train_indices]
        test_data = all_data[test_indices]

        print(train_data.shape)

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)  # Move model to GPU or CPU
        criterion.to(device)  # Move criterion to GPU or CPU
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")

        del all_data
        torch.cuda.empty_cache()

        # Example of training loop
        loss_train = 0
        epoch = 0
        num_epochs = 1
        loss_train = 1
        loss_t = 0
        num=0
        num1=0
        for step in range(train_size):  # Loop over batches (integer division)
            model.train()
            optimizer.zero_grad()

            inputs = train_data[step][0][:][:].to(
                device
            )  # Replace with your batch of images
            x = train_data[step][1][0][0]
            y = train_data[step][1][0][1]
            targets = torch.tensor([x, y], dtype=torch.float32).to(
                device
            )  # Move targets to device

            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze(0)

            # Compute loss
            loss = criterion(outputs, targets)
            loss_t = loss + loss_t
            loss.backward()
            if step % 16 == 0:
                # Backward pass and optimization

                optimizer.step()
            torch.cuda.empty_cache()

        loss_train = 0
        with torch.no_grad():
            for step in range(test_size):
                inputs = test_data[step][0][:][:].to(
                    device
                )  # Replace with your batch of images
                x = test_data[step][1][0][0]
                y = test_data[step][1][0][1]
                targets = torch.tensor([x, y], dtype=torch.float32).to(
                    device
                )  # Move targets to device

                # Forward pass
                outputs = model(inputs)
                outputs = outputs.squeeze(0)

                # Compute loss
                loss_train = criterion(outputs, targets) + loss_train
                torch.cuda.empty_cache()
            loss_train_av = loss_train / test_size
            # Print loss for each epoch
            print(loss_train_av.item())
            epoch = epoch + 1
            print(f"Epoch [{epoch}/], Loss: {loss_t / train_size}")
            torch.cuda.empty_cache()

        plt.ion()
        fig, ax = plt.subplots()
        (train_line,) = ax.plot([], [], label="train loss")
        (test_line,) = ax.plot([], [], label="test loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        losses_train, losses_test, epochs = [], [], []

        losses_train.append(int(loss_t / train_size))
        losses_test.append(int(loss_train_av.item()))
        epochs.append(epoch)

        while loss_train_av.item() >= 550:
            print("a")
            loss_t = 0
            for step in range(train_size):  # Loop over batches (integer division)
                model.train()
                optimizer.zero_grad()

                inputs = train_data[step][0][:][:].to(
                    device
                )  # Replace with your batch of images
                x = train_data[step][1][0][0]
                y = train_data[step][1][0][1]
                targets = torch.tensor([x, y], dtype=torch.float32).to(
                    device
                )  # Move targets to device

                # Forward pass
                outputs = model(inputs)
                outputs = outputs.squeeze(0)

                # Compute loss
                loss = criterion(outputs, targets)
                loss_t = loss + loss_t
                loss.backward()
                # if step % 16 ==0 :

                # Backward pass and optimization

                optimizer.step()
                torch.cuda.empty_cache()

            print("b")
            optimizer.step()

            loss_train = 0
            with torch.no_grad():
                for step in range(test_size):
                    inputs = test_data[step][0][:][:].to(
                        device
                    )  # Replace with your batch of images
                    x = test_data[step][1][0][0]
                    y = test_data[step][1][0][1]
                    targets = torch.tensor([x, y], dtype=torch.float32).to(
                        device
                    )  # Move targets to device

                    # Forward pass
                    outputs = model(inputs)
                    outputs = outputs.squeeze(0)

                    # Compute loss
                    loss_train = criterion(outputs, targets) + loss_train
                    torch.cuda.empty_cache()
                loss_train_av = loss_train / test_size
            # Print loss for each epoch
            print(loss_train_av.item())
            epoch = epoch + 1
            epochs.append(epoch)
            print(f"Epoch [{epoch}/], Loss: {loss_t / train_size}")
            a = int(loss_t / train_size)
            losses_train.append(a)
            t = int(loss_train_av.item())

            scheduler.step(t)
            losses_test.append(t)

            train_line.set_data(epochs, losses_train)
            test_line.set_data(epochs, losses_test)

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()
            if t < 400 and num ==0:
                torch.save(model.state_dict(), "custom_conv_400.pth")
            if t < 100 and num1 ==0:
                torch.save(model.state_dict(), "custom_conv_100.pth")                
            # redraw
            plt.pause(0.01)

            torch.cuda.empty_cache()

        torch.save(model.state_dict(), "custom_conv10.pth")


# Initialize and train the model
CustomTransformer()
eye_gaze_()
print("done")
'''







import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from eye_data_sorting import CombinedEyeData  # or import your updated EyeData
                                               # which now returns (img, coord) tuples
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ← UPDATED: use the new combined dataset
#from combined_dataset import CombinedEyeData  

print("a")

class CustomTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(), nn.Dropout2d(0.3), nn.MaxPool2d(2),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(), nn.Dropout2d(0.3), nn.MaxPool2d(2),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(), nn.Dropout2d(0.3), nn.MaxPool2d(2),
        )
        self.flat = nn.Flatten()

        with torch.no_grad():
            # N=1, C=1, H=210, W=160
            dummy   = torch.zeros(1, 1, 210, 160)
            n_feats = self.conv(dummy).view(1, -1).size(1)
        print(n_feats)
        # adjust these dims if your conv output size differs
        self.fin = nn.Sequential(
            nn.Linear(n_feats,256),
            nn.LeakyReLU(), nn.Dropout(0.4),
            nn.Linear(256,2),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 2:      # [H,W]
            x = x.unsqueeze(0).unsqueeze(0)  # → [1,1,H,W]
        elif x.dim() == 3:    # [C,H,W]
            x = x.unsqueeze(0)               # → [1,C,H,W]
        feats = self.conv(x)
        feats = self.flat(feats)
       # print(feats.shape)
        coords = self.fin(feats)
        return coords


class eye_gaze_(nn.Module):
    def __init__(self):
        model     = CustomTransformer()
        criterion = nn.MSELoss()
        #criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=3
        )

        # ← UPDATED: build dataset + loaders instead of manual stacking
        full_ds = CombinedEyeData(drop_last_n=0)
        train_n = int(0.90 * len(full_ds))
        test_n  = len(full_ds) - train_n
        from torch.utils.data import random_split, DataLoader
        train_ds, test_ds = random_split(full_ds, [train_n, test_n])

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        print(f"Training samples: {len(train_ds)}, Testing samples: {len(test_ds)}")

        # device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device); criterion.to(device)
        print("Using GPU" if torch.cuda.is_available() else "Using CPU")

        # clear old caches
        torch.cuda.empty_cache()

        # === initial epoch-run (mirrors your original 0→1 epoch) ===
        loss_t = 0
        for inputs, targets in train_loader:
            model.train()
            optimizer.zero_grad()
            inputs  = inputs.to(device)   # [1,1,210,210]
            targets = targets.to(device)  # [1,2]
            
            outputs = model(inputs).squeeze(0)  # → [2]
            loss    = criterion(outputs, targets.squeeze(0))
            loss_t += loss
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        # eval pass
        loss_val = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs).squeeze(0)
                loss_val += criterion(outputs, targets.squeeze(0))
                torch.cuda.empty_cache()

        loss_val_av = loss_val / len(test_ds)
        print(f"Epoch [1], Train Loss: {(loss_t/len(train_ds)).item()}, Test Loss: {loss_val_av.item()}")

        # === now your while-looped epochs with plotting/saving exactly as before ===
        plt.ion()
        fig, ax = plt.subplots()
        train_line, = ax.plot([], [], label="train loss")
        test_line,  = ax.plot([], [], label="test loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()

        epochs = [1]
        losses_train = [ (loss_t/len(train_ds)).item() ]
        losses_test  = [ loss_val_av.item() ]
        
        epoch = 1
        while loss_val_av.item() >= 550:
            epoch += 1
            loss_t = 0
            print('a')
            for inputs, targets in train_loader:
                model.train()
                optimizer.zero_grad()
                inputs  = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs).squeeze(0)
                loss    = criterion(outputs, targets.squeeze(0))
                loss_t += loss
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            loss_val = 0
            with torch.no_grad():
                print('b')
                for inputs, targets in test_loader:
                    inputs  = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs).squeeze(0)
                    loss_val += criterion(outputs, targets.squeeze(0))
                    torch.cuda.empty_cache()
            loss_val_av = loss_val / len(test_ds)

            # logging & plotting
            epochs.append(epoch)
            losses_train.append((loss_t/len(train_ds)).item())
            losses_test.append(loss_val_av.item())
            scheduler.step(loss_val_av.item())

            train_line.set_data(epochs, losses_train)
            test_line .set_data(epochs, losses_test)
            ax.relim(); ax.autoscale_view()
            fig.canvas.draw(); fig.canvas.flush_events()
            plt.pause(0.01)
            epoch=epoch+1
            print(f"Epoch {epoch}, Train Loss: {(loss_t/len(train_ds)).item()}, Test Loss: {loss_val_av.item()}")
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning rate now: {current_lr:.2e}")
            # your checkpoint logic (unchanged)
            if losses_test[-1] < 400:
                torch.save(model.state_dict(), "custom_conv_400.pth")
            if losses_test[-1] < 100:
                torch.save(model.state_dict(), "custom_conv_100.pth")

        torch.save(model.state_dict(), "custom_conv10.pth")


# run it
CustomTransformer()
eye_gaze_()
