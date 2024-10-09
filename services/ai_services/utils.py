import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np



def select_device():
    if torch.cuda.is_available(): print(f"{torch.cuda.get_device_name()} have been located and selected")
    else: print("No GPU cuda core found on this device. cpu is selected as network processor")
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class CustomResNetDataset(Dataset):
    """
    A custom dataset class for loading images and their corresponding labels for ResNet-based models.
    
    Parameters:
    img_dir (str): Path to the directory containing the image files.
    lbl_dir (str): Path to the directory containing the label files.
    transform (callable, optional): Optional transformation to be applied to the images. 
                                    If None, a default transformation (resize, normalization) is applied.
    """
    def __init__(self, img_dir:str, lbl_dir:str, transform=None):
        """
        Initializes the dataset by setting image directory, label directory, and transformations.
        
        Parameters:
        img_dir (str): Directory containing images.
        lbl_dir (str): Directory containing labels.
        transform (callable, optional): Transformations to be applied to the images. Defaults to ResNet-compatible transforms.
        """
        if transform == None: 
            transform = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.transform = transform
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.img_names = os.listdir(img_dir)
        self.lbl_names = os.listdir(lbl_dir)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
        int: The number of images in the dataset.
        """
        return len(self.img_names)
    
    def __getitem__(self, index):
        """
        Retrieve an image and its corresponding label at a specified index.
        
        Parameters:
        index (int): The index of the image and label to retrieve.

        Returns:
        image (torch.Tensor): The image after applying the transformation.
        label (tuple): A tuple containing four points corresponding to the bounding box corners.
        """
        img_path = os.path.join(self.img_dir, self.img_names[index])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        lbl_path = os.path.join(self.lbl_dir, self.lbl_names[index])
        with open(lbl_path, 'r') as file:
            raw_lbl = file.read()
        raw_lbl = raw_lbl.split(' ')[1:-2]
        raw_lbl = np.array(raw_lbl, dtype=float).reshape((4,2))
        argsorted = raw_lbl.argsort(axis=0)
        x_y = raw_lbl.copy()
        for i in range(len(argsorted.transpose())):
            x_y[argsorted.transpose()[i,:2],i] = 0
            x_y[argsorted.transpose()[i,-2:],i] = 1
        label = {}
        for i in range(len(raw_lbl)):
            if np.array_equal(x_y[i], [0., 0.]):
                label["Top Left"] = raw_lbl[i]
            elif np.array_equal(x_y[i], [0., 1.]):
                label["Top Right"] = raw_lbl[i]
            elif np.array_equal(x_y[i], [1., 1.]):
                label["Bottom Right"] = raw_lbl[i]
            elif np.array_equal(x_y[i], [1., 0.]):
                label["Bottom Left"] = raw_lbl[i]
        
        return image, label
    

# def train_custom_resnet():
#     pass
