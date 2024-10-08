import os
import cv2
import matplotlib.pylab as plt
from tqdm import tqdm


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

def read_images_from_file(file_path: str):
    """
    Read multiple images from a directory using a file path pattern.

    Parameters:
    file_path (str): A file path pattern to match images, e.g., 'images/*.jpg'.

    Returns:
    list of numpy.ndarray: A list containing the images read from the file paths.
    """
    image_paths = os.listdir(file_path)
    images_list = [None] * len(image_paths)
    for i, image_path in enumerate(tqdm(image_paths, desc="Reading images", unit="file")):
        images_list[i] = read_image(os.path.join(file_path,image_path))
    return images_list

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

def save_cropped_images(cropped_images, output_dir, name="cropped_image"):
    """
    Save cropped images to the specified directory with optional naming conventions.
    
    Parameters:
    cropped_images (list of numpy.ndarray): A list of cropped images to save. Each item should be a numpy array representing an image.
    output_dir (str): The directory where the cropped images will be saved. If the directory does not exist, it will be created.
    name (str, optional): The base name for the saved files. The default is "cropped_image". The final filename will be formed as `name_i.jpg` where `i` is the index of the cropped image. Defaults to "cropped_image".
    
    Returns:
    None: The function saves the images directly to the disk and prints the filenames of the saved images.
    
    Notes:
    - The function will create the output directory if it doesn't already exist using `os.makedirs()`.
    - The images are saved in JPEG format (`.jpg`), but the extension can be changed based on user preference or needs.
    - The `tqdm` library is used to show a progress bar while saving images, which can be helpful when dealing with a large number of images.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, cropped_image in enumerate(tqdm(cropped_images, desc="Saving images", unit="file")):
        filename = os.path.join(output_dir, f"{name}_{i}.jpg")
        cv2.imwrite(filename, cropped_image)
    print(f"Files were saved successfully in {output_dir} \n")