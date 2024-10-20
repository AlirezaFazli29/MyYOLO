import numpy as np
from services.image_handler.utils import crop_image
from torchvision.transforms import transforms
from PIL import Image
import torch
import cv2
import warnings


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
        return bb

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
    
    def batch_images(self, images, batch_size=16):
        """Split images into batches."""
        for i in range(0, len(images), batch_size):
            yield images[i:i+batch_size]  # Yield a batch of images

    def run_full_pipeline(self, images, conf=0.5, show=False, save=False, batch_size=16):
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
        
        print("Running full pipeline ... \n")
        cropped_images = []

        for i, batch in enumerate(self.batch_images(images, batch_size)):
            results = self.run_inference(batch, conf=conf, show=show, save=save)
            print(f"Yolo results are generated for batch {i} \n")
            bounding_boxes = self.extract_results_bounding_boxes(results)
            print("Found Bounding Boxes")
            batch_cropped_images = self.crop_results_bounding_boxes(bounding_boxes, batch)
            print("Images were cropped \n")
            cropped_images.extend(batch_cropped_images)
        print("All images were cropped successfully")
        return cropped_images
    

class PlateResNetInference():
    """
    A class to handle ResNet model inference and related operations.
    """
    def __init__(self, custom_resnet):
        """Initializes the inference class with a custom ResNet model.
        
        Args:
            custom_resnet: An instance of the custom ResNet model to be used for inference.
        """
        self.model = custom_resnet
        self.device = self.model.device

    def run_inference(self, batch):
        """Runs inference on a batch of images.
        
        Args:
            batch: A batch of images (torch.Tensor) to run inference on.
        
        Returns:
            results: The model's predictions for the input batch.
        """
        results = self.model.infer(batch.to(self.device))
        return results
    
    def to_pil(self, image):
        """Converts an image to a PIL Image.
        
        Args:
            image: An input image, which can be a PIL Image, numpy array, or torch Tensor.
        
        Returns:
            A PIL Image.
        
        Raises:
            TypeError: If the input is not one of the accepted types.
        """
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0)
            return Image.fromarray(image.numpy().astype(np.uint8))
        else:
            raise TypeError("Input must be a PIL Image, numpy array, or torch tensor.")

    def batch_images(self, images, transformed_images, batch_size=16):
        """Splits images into batches of a specified size.
        
        Args:
            images: A list of images to be batched.
            transformed_images: A list of transformed images corresponding to the input images.
            batch_size: The size of each batch (default is 16).
        
        Yields:
            A tuple containing a batch of images and their corresponding transformed images.
        """
        for i in range(0, len(images), batch_size):
            yield images[i:i+batch_size], transformed_images[i:i+batch_size]

    def rectify(self, image:Image.Image, corner_points:np.ndarray):
        """Applies a perspective transform to rectify the image based on detected corner points.
        
        Args:
            image: The input image to be rectified (PIL Image).
            corner_points: Detected corner points (numpy array) scaled to the image dimensions.
        
        Returns:
            rectified_image: The rectified image resized to (250, 50) pixels.
        """
        image_height, image_width = np.array(image).shape[:2]
        dst_points = np.array([[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]], dtype=np.float32)
        src_points  = corner_points * np.array([image_width, image_height])
        M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
        rectified_image = cv2.warpPerspective(np.array(image), M, (image_width, image_height))
        return cv2.resize(rectified_image, [250,50])

    def run_full_pipeline(self, images, transform=None, batch_size=16):
        """Processes images through the full inference and rectification pipeline.
        
        Args:
            images: A single image or a list of images to be processed.
            transform: An optional transformation to apply to the images before inference.
            batch_size: The size of each batch (default is 16).
        
        Returns:
            rectified_images: A list of rectified images obtained from the original images.
        """
        print("Prepairing Images ... \n")

        if isinstance(images, (np.ndarray, Image.Image, torch.Tensor)):
            # Single image pipeline
            images = [images]  # Convert single image to list format

        if transform == None: 
            transform = transforms.Compose(
                [transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
        
        transformed_images = [None] * len(images)
        for i, image in enumerate(images):
            images[i] = self.to_pil(image)
            transformed_images[i] = transform(images[i])

        print("Running full pipeline ... \n")
        rectified_images = []

        for i, batch in enumerate(self.batch_images(images ,transformed_images, batch_size)):
            imgs, imgs_t = batch
            imgs_t = torch.stack(imgs_t)
            corners = self.run_inference(imgs_t)
            corners = corners.cpu().numpy()
            corners[np.where(corners>=1)] = 1
            corners[np.where(corners<=0)] = 0
            print(f"Corner points found for batch {i} \n")
            for i, img in enumerate(imgs):
                rectified_images.append(self.rectify(img, corners[i]))
            print(f"Rectified images in batch {i} appended to the results \n")

        return rectified_images
    

class PlateUNetInference():
    """
    A class to handle UNet model inference and related operations.
    """
    def __init__(self, custom_unet):
        """Initializes the inference class with a custom UNet model.
        
        Args:
            custom_resnet: An instance of the custom ResNet model to be used for inference.
        """
        self.model = custom_unet
        self.device = self.model.device

    def run_inference(self, batch):
        """Runs inference on a batch of images.
        
        Args:
            batch: A batch of images (torch.Tensor) to run inference on.
        
        Returns:
            results: The model's predictions for the input batch.
        """
        results = self.model.infer(batch.to(self.device))
        return results
    
    def to_pil(self, image):
        """Converts an image to a PIL Image.
        
        Args:
            image: An input image, which can be a PIL Image, numpy array, or torch Tensor.
        
        Returns:
            A PIL Image.
        
        Raises:
            TypeError: If the input is not one of the accepted types.
        """
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0)
            return Image.fromarray(image.numpy().astype(np.uint8))
        else:
            raise TypeError("Input must be a PIL Image, numpy array, or torch tensor.")
        
    def batch_images(self, images, transformed_images, batch_size=16):
        """Splits images into batches of a specified size.
        
        Args:
            images: A list of images to be batched.
            transformed_images: A list of transformed images corresponding to the input images.
            batch_size: The size of each batch (default is 16).
        
        Yields:
            A tuple containing a batch of images and their corresponding transformed images.
        """
        for i in range(0, len(images), batch_size):
            yield images[i:i+batch_size], transformed_images[i:i+batch_size]

    def rectify(self, image:Image.Image, segmentation_result:np.ndarray):
        """Applies a perspective transformation to rectify the plate from the segmentation result.
        
        Args:
            image: The input image in PIL format.
            segmentation_result: The segmentation mask output from the model.
        
        Returns:
            rectified_image: The rectified image of the license plate.
        
        Raises:
            ValueError: If it fails to find the corners of the plate.
        """
        image_height, image_width = np.array(image).shape[:2]
        segmentation_result = cv2.resize(segmentation_result, (image_width, image_height))
        segmentation_result = (segmentation_result * 255).astype(np.uint8)
        _, binary = cv2.threshold(segmentation_result, 127, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        corners = None
        for i in np.arange(0.01, 0.055, 0.005):
            epsilon = i * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) == 4:
                corners = approx.squeeze(1)
                break
        if corners is None:
            warnings.warn("Could not find corners for one image in this batch. Returning original image.")
            return np.array(image) 
        sorted_corners = {
            "Top Left": list(corners[np.sqrt(np.sum((corners - np.array([0, 0])) ** 2, axis=1)).argmin()]),
            "Top Right": list(corners[np.sqrt(np.sum((corners - np.array([image_width, 0])) ** 2, axis=1)).argmin()]),
            "Bottom Right": list(corners[np.sqrt(np.sum((corners - np.array([image_width, image_height])) ** 2, axis=1)).argmin()]),
            "Bottom Left": list(corners[np.sqrt(np.sum((corners - np.array([0, image_height])) ** 2, axis=1)).argmin()])
        } 
        src_points = np.array([sorted_corners["Top Left"], 
                               sorted_corners["Top Right"], 
                               sorted_corners["Bottom Right"], 
                               sorted_corners["Bottom Left"]], dtype=np.float32)
        dst_points = np.array([[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
        rectified_image = cv2.warpPerspective(np.array(image), M, (image_width, image_height))
        return cv2.resize(rectified_image, [250,50])

    def run_full_pipeline(self, images, transform=None, batch_size=16):
        """Runs the full pipeline for inference, rectification, and batching.
        
        Args:
            images: A list of images or a single image to process.
            transform: Optional transformation to apply to the images.
            batch_size: The batch size to use for processing (default is 16).
        
        Returns:
            rectified_images: A list of rectified images for each input.
        """
        print("Prepairing Images ... \n")

        if isinstance(images, (np.ndarray, Image.Image, torch.Tensor)):
            # Single image pipeline
            images = [images]  # Convert single image to list format

        if transform == None: 
            transform = transforms.Compose(
                [transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
        
        transformed_images = [None] * len(images)
        for i, image in enumerate(images):
            images[i] = self.to_pil(image)
            transformed_images[i] = transform(images[i])

        print("Running full pipeline ... \n")

        rectified_images = []

        for i, batch in enumerate(self.batch_images(images ,transformed_images, batch_size)):
            imgs, imgs_t = batch
            imgs_t = torch.stack(imgs_t)
            plate_segments = self.run_inference(imgs_t)
            plate_segments = plate_segments.squeeze(1).cpu().numpy()
            for j, img in enumerate(imgs):
                rectified_images.append(self.rectify(img, plate_segments[j]))
            print(f"Rectified images in batch {i} appended to the results \n")

        return rectified_images
