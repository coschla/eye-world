


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
