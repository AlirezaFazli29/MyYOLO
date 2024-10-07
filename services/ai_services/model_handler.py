from abc import ABC, abstractmethod
from enum import Enum


class YoloType():
    """Enumeration for model types."""

    class Pretrained(Enum):
        yolov8n = "yolov8n.pt"
        yolov8s = "yolov8s.pt"
        yolov8m = "yolov8m.pt"
        yolov8l = "yolov8l.pt"
        yolov8x = "yolov8x.pt"

    class Custom(Enum):
        Firearm_last = "weights/last(Firearm).pt"
        Plate_last = "weights/last(plate).pt"
        Firearm_best = "weights/best(Firearm).pt"
        Plate_best = "weights/best(plate).pt"


class BaseModel(ABC):
    """
    Base class for models.

    This abstract class serves as a blueprint for all specific model implementations.
    It defines the required interface that subclasses must implement and provides
    a framework for loading and utilizing machine learning models.
    """

    def __init__(self, model_path:YoloType = YoloType.Pretrained.yolov8n):
        """
        Initializes the BaseModel with the given model path.

        Args:
            model_path (str): The file path to the model weights or configuration.
                Defaults to the pretrained YOLOv8n model.
        """
        self.model_path = model_path
        self.model = self.load_model()

    @abstractmethod
    def load_model(self):
        """Load the model (abstract method)."""
        pass  # Subclasses must implement this method

    @abstractmethod
    def infer(self, image):
        """Run inference on the given image (abstract method)."""
        pass  # Subclasses must implement this method

    def __str__(self):
        return f"Model: {self.model_path}"