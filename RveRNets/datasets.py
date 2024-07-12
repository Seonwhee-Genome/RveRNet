import os, math

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image, pad
from torchvision.io import read_image
from PIL import Image, ImageOps

class ResizeAndPad:
    def __init__(self, target_size):
        self.target_size = target_size
        
    def __call__(self, image):
        target_size = self.target_size
        w, h = image.size
        r = target_size / max(w, h)
        if r != 1:
            new_image = image.resize((math.floor(w * r), math.floor(h * r)))  # target_size를 넘지 않도록 한다. 
        else:
            new_image = image

        new_w, new_h = new_image.size
        
        h_padding = (target_size - new_w) / 2
        v_padding = (target_size - new_h) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
        
        return pad(new_image, padding, 125, 'constant')

    
class ResizeAndPadAndPermutate:
    '''
    First, resize the image and then cut the resized image into patches for patch permutation. 
    Next, add gray padding to account for the size difference between the target size and the resized image. 
    Since the size of the resized image is variable, it may not be divisible by the predetermined patch_dim. 
    Therefore, pad the resized image with black to make its size a multiple of patch_dim. 
    After the permutation is completed, apply the final padding.
    '''
    def __init__(self, target_size, patch_dim=16):
        self.target_size = target_size
        self.patch_dim = patch_dim
        
        
    def __call__(self, image):
        target_size = self.target_size        
        
        # Resize the image
        w, h = image.size
        r = target_size / max(w, h)
        if r != 1:
            new_image = image.resize((math.floor(w * r), math.floor(h * r)))
        else:
            new_image = image

        new_w, new_h = new_image.size

        # Calculate padding to make dimensions divisible by patch dimensions
        pad_w = (self.patch_dim - new_w % self.patch_dim) % self.patch_dim
        pad_h = (self.patch_dim - new_h % self.patch_dim) % self.patch_dim
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        new_image = pad(new_image, padding, fill=0)

        # Convert image to tensor
        new_image = np.array(new_image)
        new_image = torch.tensor(new_image).permute(2, 0, 1)  # Change to (C, H, W) format

        # Patch permutation
        c, h, w = new_image.shape
        shuffle_h = h // self.patch_dim
        shuffle_w = w // self.patch_dim
        patch_num = shuffle_h * shuffle_w

        inputs = rearrange(new_image, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=self.patch_dim, p2=self.patch_dim)
        row = np.random.choice(range(patch_num), size=inputs.shape[0], replace=False)
        inputs = inputs[row, :]
        inputs = rearrange(inputs, '(h w) (p1 p2 c) -> c (h p1) (w p2)',
                           h=shuffle_h, w=shuffle_w, p1=self.patch_dim, p2=self.patch_dim)
        
        # Convert tensor back to PIL image
        new_image = inputs.permute(1, 2, 0).numpy()
        new_image = Image.fromarray(new_image.astype('uint8'), 'RGB')

        # Calculate padding to reach target size
        new_w, new_h = new_image.size
        h_padding = (target_size - new_w) / 2
        v_padding = (target_size - new_h) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
        
        return pad(new_image, padding, 125, 'constant')
    
    
class ResizeAndPadAndTranslocate:
    def __init__(self, target_size, dx=60, dy=85):
        self.target_size = target_size
        self.dx = dx
        self.dy = dy
        
    def __call__(self, image):
        target_size = self.target_size
        dx = self.dx
        dy = self.dy
        
        # Resize the image
        w, h = image.size
        r = target_size / max(w, h)
        if r != 1:
            new_image = image.resize((math.floor(w * r), math.floor(h * r)))  # target_size를 넘지 않도록 한다.
        else:
            new_image = image
        
        # Translocate the image
        translocated_image = Image.new('RGB', (target_size, target_size), (125, 125, 125))
        translocated_image.paste(new_image, (dx, -dy))
        # Crop the image to target size
        final_image = translocated_image.crop((0, 0, target_size, target_size))
        
        return final_image  
    

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, label_to_int, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []
        self.img_paths = []
        
        self.label_to_int = label_to_int  # use the predefined dictionary
        self.classList = list(label_to_int.keys())

        for class_name in self.classList:
            class_dir = os.path.join(img_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.img_labels.append(class_name)
                    self.img_paths.append(os.path.join(class_dir, img_name))


    def __len__(self):
        return len(self.img_paths)
    

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)        
        image = to_pil_image(image)
        label = self.img_labels[idx]
        if self.transform:         
            image = self.transform(image)
        
        try:
            label_idx = self.label_to_int.get(label, -1)  # Get label index, default to -1 if not found

        except KeyError:
            raise KeyError(f"The label '{label}' at index {idx} was not found in the label mapping.")
        
        label_tensor = torch.tensor(label_idx, dtype=torch.long)            

        return image, label_tensor, os.path.basename(img_path)  # return image, label, and file name


class CustomImageDataset_inference(Dataset):
    '''
    img_dir/
      |--crop/
      |--mask/
      |__yolo/
    '''
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform        
        self.img_paths = []        
        
        if os.path.isdir(img_dir):
            for img_name in os.listdir(img_dir):
                self.img_paths.append(os.path.join(img_dir, img_name))
                
                
    def __len__(self):
        return len(self.img_paths)
    

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = os.path.basename(img_path).strip(".jpg")          
        image = read_image(img_path)        
        image = to_pil_image(image)    
        if self.transform:         
            image = self.transform(image)         

        return image, img_name#, coord  # return image, and the filename without the extension
    

class PairedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.name_to_idx = {file_name: idx for idx, (img, label, file_name) in enumerate(dataset2)}
        
        
    def __len__(self):
        return len(self.dataset1)
    

    def __getitem__(self, idx):
        img1, label1, name1 = self.dataset1[idx]
        idx2 = self.name_to_idx.get(name1)
        if idx2 is not None:
            img2, label2, name2 = self.dataset2[idx2]
            if name1 != name2:
                raise ValueError(f"Mismatched labels for {name1}: {label1.item()} vs {label2.item()}")
        else:
            raise KeyError(f"No matching file found for {name1} in the second dataset")
        return img1, img2, label1, name1


class PairedDataset_inference(PairedDataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.name_to_idx = {file_name: idx for idx, (img, file_name) in enumerate(dataset2)}  


    def __getitem__(self, idx):
        img1, name1 = self.dataset1[idx]
        idx2 = self.name_to_idx.get(name1)
        if idx2 is not None:
            img2, name2 = self.dataset2[idx2]
            if name1 != name2:
                raise ValueError(f"Mismatched labels for {name1}: {label1.item()} vs {label2.item()}")
        else:
            raise KeyError(f"No matching file found for {name1} in the second dataset")
        return img1, img2, name1  