from .model_handler import BaseModel, YoloType
from ultralytics import YOLO
from torchinfo import summary


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

        print(f"Loading YOLO model from {self.model_path} ...")
        if isinstance(self.model_path, (YoloType.Pretrained ,YoloType.Custom)):
            yolo = YOLO(model=self.model_path.value)
        else:
            raise ValueError("Invalid model type. Must be YoloType.Pretrained or YoloType.Custom.")
        print('\n', summary(yolo))
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