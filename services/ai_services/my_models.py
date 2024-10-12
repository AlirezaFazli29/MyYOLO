from .model_handler import BaseModel, YoloType, ResNetType
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary
from .utils import train_custom_resnet
import time



class YOLOv8(BaseModel):
    """Class for handling YOLOv8 model operations."""

    def load_model(self):
        """
        Load the YOLO model.
        
        This method initializes the YOLO model using the specified model path. 
        It checks if the model path is a valid type (either a pretrained model or a custom model).
        If valid, it loads the model and prints a summary of the loaded model.
        If invalid, it raises a ValueError.

        Returns:
            YOLO: The initialized YOLO model instance.
        """

        print(f"Loading YOLO model from {self.model_path.value} ...")
        if isinstance(self.model_path, (YoloType.Pretrained ,YoloType.Custom)):
            yolo = YOLO(model=self.model_path.value)
        else:
            raise ValueError("Invalid model type. Must be YoloType.Pretrained or YoloType.Custom.")
        print(f"YOLO: {self.model_path.value}")
        print(summary(yolo), '\n')
        return yolo

    def infer(self, image, show:bool=False, conf:float=0.5, save:bool=False):
        """
        Run inference on the given image.
        
        This method uses the loaded YOLO model to run inference on a specified image.
        It provides options to show the results, set the confidence threshold, and save the output.

        Args:
            image: The input image for inference (can be a file path or image array).
            show (bool): Flag indicating whether to display the results (default is False).
            conf (float): Confidence threshold for detections (default is 0.5).
            save (bool): Flag indicating whether to save the output results (default is False).
        
        Returns:
            list: The results of the inference, containing detected objects and their details.
        """

        print("Running inference on Image")
        results = self.model(source=image, show=show, conf=conf, save=save)
        return results
    

class CustomResNet(nn.Module):
    """
    A custom ResNet18-based model that allows for dynamic classifier modification 
    and freezing/unfreezing of the feature extractor (ResNet18 backbone).
    """
    def __init__(self, weights=models.ResNet18_Weights.IMAGENET1K_V1):
        super(CustomResNet, self).__init__()
        """
        Initialize the CustomResNet model with a ResNet18 backbone and custom fully connected layers.
        
        Parameters:
        weights (torchvision.models.ResNet18_Weights, optional): Pretrained weights for the ResNet18 model. 
                                                                  Default is IMAGENET1K_V1.
        """
        self.feature = models.resnet18(weights=weights)
        self.feature.fc = nn.Dropout(0.2)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=512, out_features=128), 
            nn.SiLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.SiLU()
        )
        self.fc0 = nn.Linear(in_features=32, out_features=2)
        self.fc1 = nn.Linear(in_features=32, out_features=2)
        self.fc2 = nn.Linear(in_features=32, out_features=2)
        self.fc3 = nn.Linear(in_features=32, out_features=2)
        
    def forward(self, x):
        x = self.feature(x)
        x = self.mlp(x)
        x0y0 = self.fc0(x)
        x1y1 = self.fc1(x)
        x2y2 = self.fc2(x)
        x3y3 = self.fc3(x)
        return torch.stack([x0y0, x1y1, x2y2, x3y3], dim=1)


class Plate_ResNet(BaseModel):
    """
    A custom ResNet-based model for plate detection or similar tasks.
    This class supports different ResNet model types (e.g., Base, Custom) and can run inference 
    on image batches.
    """
    def __init__(self,
                 model_type:ResNetType=ResNetType.Base,
                 weigths:models.ResNet18_Weights=models.ResNet18_Weights.IMAGENET1K_V1,
                 device='cpu'):
        """
        Initialize the Plate_ResNet model with a specific ResNet type, pretrained weights, and device.

        Parameters:
        model_type (ResNetType, optional): The type of ResNet model to use (e.g., Base, Custom). Default is Base.
        weigths (torchvision.models.ResNet18_Weights, optional): Pretrained weights for the model. Default is ResNet18 weights from ImageNet.
        device (str, optional): The device on which to run the model (e.g., 'cpu', 'cuda'). Default is 'cpu'.
        """
        self.model_type = model_type
        self.weigths = weigths
        self.device = device
        self.model = self.load_model()
        self.model_training_history = { 'train_loss' : [], 'valid_loss': []}

    def load_model(self):
        """
        Load the ResNet model with custom modifications based on the selected model type and pretrained weights.
        
        Returns:
        torch.nn.Module: The loaded ResNet model ready for inference.
        """
        resnet = CustomResNet(weights=self.weigths).to(self.device)
        print(f"Selected model is {self.model_type}")
        if self.model_type != ResNetType.Base:
            resnet.load_state_dict(torch.load(self.model_type.value, 
                                              map_location=torch.device(self.device), 
                                              weights_only=True))
        print("ResNet18")
        print(summary(resnet), '\n')
        return resnet
        
    def infer(self, img_batch):
        """
        Run inference on a batch of images.

        Parameters:
        img_batch (torch.Tensor): A batch of input images to run inference on.

        Returns:
        tuple: Inference results from the model.
        """
        print("Running inference on Image Batch")
        if self.model.training:
            self.model.eval()
        img_batch = img_batch.to(self.device)
        with torch.no_grad():
            results = self.model(img_batch)
        print("Outputs Generated")
        return results

    def freeze_unfreeze(self, freeze:bool=False):

        print(f"Start {'freezing' if freeze==True else 'unfreezing'} feature extractor ...")
        for param in self.model.feature.parameters():
            param.requires_grad = not(freeze)
        print(f"Feature extractor {'freezed' if freeze==True else 'unfreezed'}")

    def write_summary(self):
        print(f"Model is in {self.device}")
        print(summary(self.model))

    def train(
        self,
        train_loader,
        valid_loader,
        optimizer=None,
        lr:float=0.01,
        weight_decay=0.001,
        epochs:int=10,
        loss_fn=torch.nn.MSELoss(),
        save_parameters=True,
        save_path:str="./weights"
    ):
        """
        Train the ResNet model using the provided training and validation loaders.

        Parameters:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer, optional): Optimizer for model parameters. If None, a default optimizer is used.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.
        weight_decay (float, optional): Weight decay (L2 regularization) parameter for the optimizer. Defaults to 0.001.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        loss_fn (torch.nn.Module, optional): Loss function used for training. Defaults to MSELoss.
        save_parameters (bool, optional): Flag to save model parameters. Defaults to True.
        save_path (str, optional): Path to save model weights. Defaults to './weights'.

        Returns:
        None
        """
        print(f"Starting training process for model {self.model_type} ...")
        print(f"Hyper Parameters:")
        print(f"- Epochs Number = {epochs}")
        print(f"- Learning Rate = {lr}")
        print(f"- Weight Decay = {weight_decay}")
        print(f"- Loss Function = {loss_fn}")
        print(f"- Optimizer = {torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay) if None else optimizer} \n")
        start_time = time.time()
        training_history = train_custom_resnet(
            model=self.model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            lr=lr,
            epochs=epochs,
            loss_fn=loss_fn,
            device=self.device,
            save_parameters=save_parameters,
            save_path=save_path
        )
        print("\nTraining completed")
        end_time = time.time()
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        print(f"Training time: {minutes} minutes and {seconds:.2f} seconds")
        self.model_training_history["train_loss"].extend(training_history["train_loss"])
        self.model_training_history["valid_loss"].extend(training_history["valid_loss"])