import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import os



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
        label = {
            "Top Left": [0., 0.],
            "Top Right": [223., 0.],
            "Bottom Right": [223., 223.],
            "Bottom Left": [0., 223.]
        }
        for i in range(len(raw_lbl)):
            if np.array_equal(x_y[i], [0., 0.]):
                label["Top Left"] = list(raw_lbl[i])
            elif np.array_equal(x_y[i], [1., 0.]):
                label["Top Right"] = list(raw_lbl[i])
            elif np.array_equal(x_y[i], [1., 1.]):
                label["Bottom Right"] = list(raw_lbl[i])
            elif np.array_equal(x_y[i], [0., 1.]):
                label["Bottom Left"] = list(raw_lbl[i])
        
        return image, torch.tensor([label["Top Left"], 
                                    label["Top Right"], 
                                    label["Bottom Right"], 
                                    label["Bottom Left"]], dtype=torch.float32)
    

def train_custom_resnet(
        model,
        train_loader,
        valid_loader,
        optimizer=None,
        lr:float=0.01,
        weight_decay=0.01,
        epochs:int=10,
        loss_fn=torch.nn.MSELoss(),
        device='cpu',
        save_parameters=True,
        save_path:str="./weights"
    ):
    """
    Train a custom ResNet model with specified parameters.

    Parameters:
    model (torch.nn.Module): The model to be trained.
    train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    optimizer (torch.optim.Optimizer, optional): Optimizer for model parameters. Defaults to Adam with lr.
    lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.
    weight_decay (float, optional): Weight decay (L2 regularization) parameter for the optimizer. Defaults to 0.01.
    epochs (int, optional): Number of training epochs. Defaults to 10.
    loss_fn (torch.nn.Module, optional): Loss function used for training. Defaults to MSELoss.
    device (str, optional): Device for model training (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
    save_parameters (bool, optional): Flag to save model parameters. Defaults to True.
    save_path (str, optional): Path to save model weights. Defaults to './weights'.
    
    Returns:
    dict: A dictionary containing training and validation loss history.
    """

    model.to(device)
    optimizer = optimizer or torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)

    history = { 'train_loss' : [], 'valid_loss': []}
    min_loss = 1e9

    
    for ep in range(epochs):

        # Training Phase
        model.train()

        steps = train_loader.__len__()
        total_loss, count = 0, 0

        for feature, labels in tqdm(train_loader, total=steps, desc= f'Training Epoch {ep+1:2}', leave=False):
            imgs, lbls = feature.to(device), labels.to(device)
            optimizer.zero_grad()   
            out = model(imgs)   
            loss = loss_fn(out, lbls)   
            loss.backward()   
            optimizer.step()
            total_loss += loss
            count += len(lbls)

        # Calculate average training loss
        training_loss = total_loss.item()/count
        history["train_loss"].append(training_loss)

        # Validation Phase
        model.eval()

        steps = valid_loader.__len__()
        total_loss, count = 0, 0

        with torch.no_grad():
            for feature, labels in tqdm(valid_loader, total=steps, desc= f'Validating Epoch {ep+1:2}', leave=False):         
                imgs, lbls = feature.to(device), labels.to(device)
                out = model(imgs)
                loss = loss_fn(out, lbls)
                total_loss += loss
                count += len(lbls)

        # Calculate average validation loss
        validation_loss = total_loss.item()/count
        history["valid_loss"].append(validation_loss)

        print(f"Epoch {ep:2}: Train loss={training_loss:.3f}, Val loss={validation_loss:.3f}")

        if save_parameters:
            torch.save(model.state_dict(), os.path.join(save_path, "last_custom_res.pth"))
            if validation_loss<=min_loss:
              min_loss = validation_loss
              torch.save(model.state_dict(), os.path.join(save_path, "best_custom_res.pth"))

    return history