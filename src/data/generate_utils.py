import cv2
import numpy as np
from zipfile import ZipFile
from PIL import Image

def save2zip(img: np.array, img_filename: str, filename: str, path:str) -> None:
    """Save image to the .zip file

    :param img: 2D array which represents an image
    :type img: np.array
    :param img_filename: Name of the image file
    :type img_filename: str
    :param filename: Name of the .zip file
    :type filename: str
    :param path: Path to the directory where .zip file is stored
    :type path: str
    """
    zip_path = filename
    if path:
        zip_path = path+filename

    _, encoded_image = cv2.imencode(".png", img)
    with ZipFile(zip_path, "a") as zip:
        zip.writestr(img_filename, encoded_image.tobytes())
        
def save2directory(img: np.array, img_filename: str, path: str) -> None:
    """Save image to the directory

    :param img: 2D array which represents an image
    :type img: np.array
    :param img_filename: Name of the image file
    :type img_filename: str
    :param path: Path to the output directory
    :type path: str
    """
    img = Image.fromarray(img)
    img.save(path+img_filename)