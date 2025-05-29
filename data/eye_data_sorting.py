'''
from eye_gaze_text2 import read_gaze_data_txt2
from eye_gaze_text import read_gaze_data_txt
from image import CustomImageDataset
from image_2 import CustomImageDataset2
import torch
from torch.utils.data import Dataset  # , DataLoader
from torchvision import transforms
import numpy as np

class EyeData(Dataset):
    def __init__(self):
        # Define the transforms: Grayscale and Padding
        self.to_pil = (
            transforms.ToPILImage()
        )  # Converts tensor to PIL Image for transformation
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(
                    num_output_channels=1
                ),  # Convert image to grayscale
                transforms.Pad(
                    (0, 0, 0, 0)
                ),  # Padding to make the image 210x210 (if smaller)
                transforms.ToTensor(),  # Convert to tensor
            ]
        )

        # Load the image-only Dataset
        self.raw = CustomImageDataset()
        # Read gaze data: list of dicts with key "gaze_positions"
        self.gaze = read_gaze_data_txt(
            r"data\eye_gaze_root\52_RZ_2394668_Aug-10-14-52-42.txt"
        )

        # Drop the last 200 if needed
        self.num_samples = len(self.raw) 

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1) Load image (may return a single Tensor or a tuple)
        raw_item = self.raw[idx]
        if isinstance(raw_item, (list, tuple)) and len(raw_item) >= 1:
            img = raw_item[0]
        else:
            img = raw_item

        # Convert the tensor to a PIL Image before applying transformations
        img = self.to_pil(img)

        # Apply the transformations (Grayscale and Padding)
        img = self.transform(img)

        # 2) Grab the last gaze coordinate (use .iloc for row access)
        gaze_entry = self.gaze.iloc[idx]  # Use .iloc to access the row by index
        gaze_positions = gaze_entry.get("gaze_positions", [])

        if not gaze_positions:
            return None  # Skip this sample if there are no gaze positions

        x, y = gaze_positions[-1]

        # 3) Build a combined tensor for visualization/debug
        data = torch.zeros(2, 210, 160)  # One additional channel for the coordinates
        data[0] = img  # Store the image in the first channel
        data[1, 0, 0] = x  # Store the x coordinate in a specific position
        data[1, 0, 1] = y  # Store the y coordinate in a specific position

        # 4) Return the combined image and target coordinate tensor
        return data


# Now, to collect all the samples in tensors for batching
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

# Print the final shape of the dataset (all images and coordinates combined)
#print(f"All data shape (num_samples, 1, 210, 210): {all_data.shape}")

# Now, split the data into training and testing groups (85% training, 15% testing)
train_size = int(0.85 * len(all_data))
test_size = len(all_data) - train_size

# Shuffle the dataset to randomize the split
indices = np.random.permutation(len(all_data))

train_indices, test_indices = indices[:train_size], indices[train_size:]

# Create the training and testing datasets
train_data = all_data[train_indices]
test_data = all_data[test_indices]

# Print the size of the training and testing sets
print(f"Training data size: {train_data.shape}")
print(f"Testing data size: {test_data.shape}")
"""
# Optionally, create DataLoaders for batching
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Example of iterating through the train_loader
for batch_data in train_loader:
    images = batch_data[:, 0, :, :]  # Extract the images from the batch
    coordinates = batch_data[:, 1, 0, :]  # Extract the coordinates from the batch
    print(f"Batch images shape: {images.shape}, Batch coordinates shape: {coordinates.shape}")
    break  # Just to demonstrate the first batch
"""
'''


'''

class CombinedEyeData(Dataset):
    def __init__(self, drop_last_n: int = 0):
        # 1) Prepare transforms (pad width to 210, then ToTensor→float32/0–1)
        self.to_pil = transforms.ToPILImage()
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Pad((0, 0, 0, 0)),  # widen 160→210
            transforms.ToTensor(),
        ])

        # 2) Load both image datasets
        self.raw1 = CustomImageDataset()
        self.raw2 = CustomImageDataset2()

        # 3) Load both gaze-dataframes
        self.gaze1 = read_gaze_data_txt( 
            r"data/eye_gaze_root/52_RZ_2394668_Aug-10-14-52-42.txt"
        )
        self.gaze2 = read_gaze_data_txt2(
            r"other_data/eye_gaze_root2/52_RZ_2394668_Aug-10-14-52-42.txt"
        )

        # 4) Optionally drop last N samples from each
        if drop_last_n > 0:
            self.raw1 = torch.utils.data.Subset(self.raw1, list(range(len(self.raw1)-drop_last_n)))
            self.gaze1 = self.gaze1.iloc[:len(self.raw1)]
            self.raw2 = torch.utils.data.Subset(self.raw2, list(range(len(self.raw2)-drop_last_n)))
            self.gaze2 = self.gaze2.iloc[:len(self.raw2)]

        # 5) store lengths
        self.len1 = len(self.raw1)
        self.len2 = len(self.raw2)

    def __len__(self):
        # total = both sources
        return self.len1 + self.len2

    def __getitem__(self, idx):
        # pick source
        if idx < self.len1:
            raw_item  = self.raw1[idx]
            gaze_row  = self.gaze1.iloc[idx]
        else:
            raw_item  = self.raw2[idx - self.len1]
            gaze_row  = self.gaze2.iloc[idx - self.len1]

        # unpack image
        if isinstance(raw_item, (list, tuple)):
            img = raw_item[0]
        else:
            img = raw_item

        # PIL→transform
        img = self.to_pil(img)
        img = self.transform(img)   # → [1,210,210]

        # grab last gaze point
        gaze_positions = gaze_row.get("gaze_positions", [])
        if not gaze_positions:
            # could skip but then length logic gets tricky—return dummy
            return img, torch.tensor([0.,0.], dtype=torch.float32)

        x, y = gaze_positions[-1]
        coord = torch.tensor([x, y], dtype=torch.float32)

        return img, coord

'''



from eye_gaze_text  import read_gaze_data_txt
#from eye_gaze_text2 import read_gaze_data_txt2
from image        import CustomImageDataset
from image_2      import CustomImageDataset2
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms 
from torch.utils.data import random_split, DataLoader
import webdataset 
import json
import io

class CombinedEyeData(Dataset):
    def __init__(self, drop_last_n: int = 0):
        # 1) Prepare transforms (pad width to 210, then ToTensor→float32/0–1)
        self.to_pil = transforms.ToPILImage()
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Pad((0, 0, 0, 0)),  # widen 160→210
            #transforms.ToTensor(),
        ])

        # 2) Load both image datasets
        self.raw1 = CustomImageDataset()
        self.raw2 = CustomImageDataset2()

        # 3) Load both gaze-dataframes
        self.gaze1 = read_gaze_data_txt( 
            r"data/eye_gaze_root/52_RZ_2394668_Aug-10-14-52-42.txt"
        )

        # 4) Optionally drop last N samples from each
        if drop_last_n > 0:
            self.raw1 = torch.utils.data.Subset(self.raw1, list(range(len(self.raw1)-drop_last_n)))
            self.gaze1 = self.gaze1.iloc[:len(self.raw1)]


        # 5) store lengths
        self.len1 = len(self.raw1)


    def __len__(self):
        # total = both sources
        return self.len1 

    def __getitem__(self, idx):
        # pick source
        if idx < self.len1:
            raw_item  = self.raw1[idx]
            gaze_row  = self.gaze1.iloc[idx]


        # unpack image
        if isinstance(raw_item, (list, tuple)):
            img = raw_item[0]
        else:
            img = raw_item

        # PIL→transform
        img = self.to_pil(img)
        img = self.transform(img)   # → [1,210,210]

        # grab last gaze point
        gaze_positions = gaze_row.get("gaze_positions", [])
        if not gaze_positions:
            # could skip but then length logic gets tricky—return dummy
            return img, torch.tensor([0.,0.], dtype=torch.float32)

        x, y = gaze_positions[-1]
        coord = torch.tensor([x, y], dtype=torch.float32)

        return img, coord
    


class main():
    def __init__(self):


        full_ds = CombinedEyeData(drop_last_n=0)
        test_n  = len(full_ds)
        print(full_ds)
        
        with webdataset.TarWriter("dataset__pred_1.tar") as writer:
            for x in range(test_n):

                img, coord = full_ds[x]
                #img_tensor = img.detach().cpu()

                # 1) serialize the tensor itself
                buf = io.BytesIO()
                torch.save(img, buf)
                tensor_bytes = buf.getvalue()

                # 2) serialize the coords (e.g. a tensor or list)
                #    here we turn it into JSON text
                coord_list = coord.tolist() #if hasattr(coord, "tolist") else coord
                coord= json.dumps(coord_list).encode("utf-8")


                sample = {

                    "__key__" : str(x),
                    "img" :img,
                    "coord":coord
                }

                writer.write(sample)
        
main()
print('f')