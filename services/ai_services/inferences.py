import numpy as np
from services.image_handler.utils import crop_image


class YOLOInference():
    """
    A class to handle YOLO model inference and related operations such as extracting bounding boxes
    and cropping images based on bounding box coordinates.
    """

    def __init__(self, yolo_model):
        """
        Initialize the YOLOInference class with a YOLO model.
        
        Parameters:
        yolo_model (BaseModel): An instance of a YOLO model to use for inference.
        """
        self.yolo_model = yolo_model

    def run_inference(self, image, conf=0.5, show=False, save=False):
        """
        Run YOLO model inference on an image.
        
        Parameters:
        image (numpy.ndarray): The input image on which to run inference.
        conf (float, optional): Confidence threshold for predictions. Defaults to 0.5.
        show (bool, optional): If True, displays the image with bounding boxes. Defaults to False.
        save (bool, optional): If True, saves the inference results. Defaults to False.
        
        Returns:
        ultralytics.yolo.engine.results.Results: The inference results, including bounding boxes and confidence scores.
        """
        print(f"conf={conf}, show={show}, save={save}")
        results = self.yolo_model.infer(image, show=show, conf=conf, save=save)
        return results

    def extract_image_bounding_boxes(self, result):
        """
        Extract bounding boxes from a single inference result.

        Parameters:
        result (ultralytics.yolo.engine.results.Result): A single YOLO inference result containing bounding boxes.

        Returns:
        numpy.ndarray: An array of bounding boxes with coordinates in the format [x1, y1, x2, y2].
        """
        bb = result.boxes  
        bb = np.array(bb.xyxy.cpu(), dtype=int)
        return bb

    def extract_results_bounding_boxes(self, results):
        """
        Extract bounding boxes from multiple YOLO inference results.

        Parameters:
        results (list of ultralytics.yolo.engine.results.Result): A list of YOLO inference results.

        Returns:
        numpy.ndarray: A 2D array where each entry contains bounding box coordinates from one result.
        """
        bb = [None] * len(results)
        for i, result in enumerate(results):
            bb[i] = self.extract_image_bounding_boxes(result)
        return np.array(bb)

    def crop_image_bounding_boxes(self, bounding_box_array, image):
        """
        Crop an image using a list of bounding boxes.

        Parameters:
        bounding_box_array (numpy.ndarray): An array of bounding box coordinates.
        image (numpy.ndarray): The input image to be cropped.

        Returns:
        list of numpy.ndarray: A list of cropped image sections corresponding to each bounding box.
        """
        cropped = [None]*len(bounding_box_array)
        for i,bb in enumerate(bounding_box_array):
            cropped[i] = crop_image(image, bb)
        return cropped


    def crop_results_bounding_boxes(self, results_bounding_boxes, images):
        """
        Crop multiple images using bounding boxes from YOLO inference results.

        Parameters:
        results_bounding_boxes (list of numpy.ndarray): A list of bounding box arrays, one for each image.
        images (list of numpy.ndarray): A list of input images corresponding to the bounding box arrays.

        Returns:
        list of numpy.ndarray: A flattened list of all cropped image sections for each input image.
        """
        cropped_images = []
        for i,image in enumerate(images):
            cropped = self.crop_image_bounding_boxes(results_bounding_boxes[i], image)
            if i==0: cropped_images = cropped
            else: cropped_images += cropped
        return cropped_images
    
    def run_full_pipeline(self, images, conf=0.5, show=False, save=False):
        """
        Run the full pipeline: inference, bounding box extraction, and cropping images.
        
        Parameters:
        images (numpy.ndarray or list of numpy.ndarray): The input image or list of images to process.
        conf (float, optional): Confidence threshold for predictions. Defaults to 0.5.
        show (bool, optional): If True, displays the image with bounding boxes. Defaults to False.
        save (bool, optional): If True, saves the inference results. Defaults to False.
        
        Returns:
        list of numpy.ndarray: A list of cropped image sections based on bounding box coordinates.
        """
        if isinstance(images, np.ndarray):
            # Single image pipeline
            images = [images]  # Convert single image to list format

        print("Running full pipeline... \n")
        results = self.run_inference(images, conf=conf, show=show, save=save)
        print("Yolo results are generated \n")
        bounding_boxes = self.extract_results_bounding_boxes(results)
        print("Found Bounding Boxes")
        cropped_images = self.crop_results_bounding_boxes(bounding_boxes, images)
        print("Images are cropped \n")
        return cropped_images