import cv2
import numpy as np
import random
from tqdm import tqdm

import os
import pkg_resources

from src.data.generate_utils import save2directory, save2zip

def _check_args(num_images: int, num_used_raw_image: int):
    if num_images <= 0:
        raise ValueError("Number of generated images must be grater than 0.")
    if num_used_raw_image <= 20:
        raise ValueError("Number of images used to extract noise must be greater than 20.")
    
    
def load_random_sample_filename() -> str:
    """Load random filename of the raw image from package samples

    :return: Filename
    :rtype: str
    """
    raw_img_filename = random.randint(0, 299)
    raw_img_filename = f"{str(raw_img_filename).zfill(5)}.png"
    file = pkg_resources.resource_filename(__name__, f"/samples/raw/{raw_img_filename}")
    return file
    
def load_random_dir_filename(path: str) -> str:
    """Load random filename from given directory

    :param path: Path to the directory
    :type path: str
    :return: Filename
    :rtype: str
    """
    return path+random.choice(os.listdir(path))

def generate_random_noise(num_used_raw_images: int=100, path_to_raw: str=None) -> np.array:
    """Generate random noise image which will be added to artificially generated pure images.
    Method randomly takes num_images raw frames and extracts noise from them.

    :param num_used_raw_images: Number of images that are used to create one noise image, defaults to 100
    :type num_used_raw_images: int, optional
    """
    noise_image = np.zeros((480,640))
    for _ in range(num_used_raw_images):
        if path_to_raw:
            noise_filename = load_random_dir_filename(path_to_raw)
        else:
            noise_filename = load_random_sample_filename()
        img = cv2.imread(noise_filename)
        img = img[4:484,4:644,0]
        noise_image = (noise_image+img)
    noise_image = noise_image/num_used_raw_images
    return noise_image.astype(np.uint8)


def generate_noise_dataset( path: str, 
                            num_images: int=50, 
                            num_used_raw_images: int=100, 
                            path_to_raw: str=None,
                            zipfile: bool=False,
                            zip_filename: str=None) -> None:
    """Generate random noise images

    :param num_images: Number of the generated noise images, defaults to 50
    :type num_images: int, optional
    :param num_used_raw_images: Number of images that are used to create one noise image, defaults to 100
    :type num_used_raw_images: int, optional
    """
    _check_args(num_images, num_used_raw_images)
    for frame in tqdm(range(num_images)):
        noise_image = generate_random_noise(num_used_raw_images, path_to_raw)
        
        if zipfile:
            save2zip(noise_image, img_filename=f"{frame}.png", filename=zip_filename, path=path)
        else:
            save2directory(noise_image, img_filename=f"{frame}.png", path=path)




if __name__ == "__main__":
    pass
