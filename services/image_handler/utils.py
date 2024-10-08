import cv2
import numpy as np
import matplotlib.pylab as plt


def read_image(image_path: str):
    """
    Read an image from a file.
    
    Parameters:
    image_path (str): Path to the image file.

    Returns:
    image (numpy.ndarray): Image read from the file, represented as a NumPy array.
    """
    image = cv2.imread(image_path)
    return image

def convert_to_rgb(image):
    """
    Convert a BGR image to RGB format.
    
    Parameters:
    image (numpy.ndarray): Image in BGR format (as read by OpenCV).

    Returns:
    numpy.ndarray: Image converted to RGB format.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_to_gray(image):
    """
    Convert an image to grayscale.
    
    Parameters:
    image (numpy.ndarray): Input image (in RGB or BGR format).

    Returns:
    numpy.ndarray: Grayscale version of the input image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def crop_image(image, bbox):
    """
    Crop an image using the provided bounding box coordinates.
    
    Parameters:
    image (numpy.ndarray): Input image to be cropped.
    bbox (tuple): A tuple (x1, y1, x2, y2) representing the bounding box coordinates. 
                  x1, y1 = top-left corner, x2, y2 = bottom-right corner.

    Returns:
    numpy.ndarray: Cropped image.
    """
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]

def show_image(image, turn_grey=False,cmap=None):
    """
    Display an image using Matplotlib.
    
    Parameters:
    image (numpy.ndarray): The image to display.
    turn_grey (bool, optional): If True, the image will be converted to grayscale before displaying.
                                Defaults to False.
    cmap (str, optional): Colormap for displaying grayscale images (e.g., 'gray'). Defaults to None, 
                          which shows the image in color.
    
    Returns:
    None: Displays the image using Matplotlib and hides the axis.
    """
    if turn_grey==True: image = convert_to_gray(image)

    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()