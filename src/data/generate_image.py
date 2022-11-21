import cv2
import numpy as np
from typing import Tuple
import pkg_resources

from src.data._models import ImageDetails

def _check_image(size: Tuple[int, int], epsilon: float, ring_center: Tuple[int, int], brightness: Tuple[int, int], noise_file_index) -> None:
    width, height = size
    width_ring_center, height_ring_center = ring_center
    min_brightness, max_brightness = brightness
    _ = ImageDetails(width=width,
              height=height,
              epsilon=epsilon,
              ring_center_width=width_ring_center,
              ring_center_height=height_ring_center,
              min_brightness=min_brightness,
              max_brightness=max_brightness,
              used_noise=noise_file_index)

def generate_pure_image(size: Tuple[int, int],
                        epsilon: float, 
                        ring_center: Tuple[int, int], 
                        brightness: Tuple[int, int]) -> np.array:
    """Generate pure image

    :param size: Size of the image (width, height)
    :type size: Tuple[int, int]
    :param epsilon: Epsilon value
    :type epsilon: float
    :param ring_center: Position of the central ring
    :type ring_center: Tuple[int, int]
    :param brightness: Range of brightness
    :type brightness: Tuple[int, int]
    :return: 2D array which represents pure image
    :rtype: np.array
    """
    width, height = size
    min_brightness, max_brightness =  brightness
    
    mean_brightness = (min_brightness + max_brightness) / 2
    diff_brightness = max_brightness - mean_brightness
    
    diff_betweeen_rings_denominator = 6.07
    diff_between_rings = width*width / diff_betweeen_rings_denominator
    
    width_ring_center, height_ring_center = ring_center
    
    y, x = np.indices([height, width])
    img = mean_brightness + (diff_brightness * np.cos(2*np.pi*(1.0-epsilon + ((np.power((x-width_ring_center)*2, 2) + np.power((y-height_ring_center)*2, 2)) / diff_between_rings))))
    img = img.astype(np.uint8)
    return img

def add_noise_to_image(pure_image: np.array, noise: np.array) -> np.array:
    """Add random noise to the pure image

    :param pure_image: 2D array which represents pure image
    :type pure_image: np.array
    :param noise: 2D array which represents noise image
    :type noise: np.array
    :return: Noised image
    :rtype: np.array
    """
    noise = noise[:,:,0]
    noise_mean = np.mean(noise)
    difference = -(noise-noise_mean)
    noised_image = pure_image-difference
    noised_image = np.clip(noised_image, 0, 255)
    return noised_image

def generate_image(epsilon: float,
                   size: Tuple[int, int]=(640, 480),
                   ring_center: Tuple[int, int]=(320, 240),
                   brightness: Tuple[int, int]=(80, 210),
                   noise_path: str=None,
                   noise_file_index: int=0
                    ) -> np.array:
    """Generate the image. In case of generating single image (using this function) you don't have to pass noise_path argument only if you use this code as a package.
    If you didn't install it via pip, you have to pass the argument noise_path. It is assumed that noise images in you directory are named with integers started from
    0 to N, e.g. you have 5 noise image in you directory, so files are named: 0.png, 1.png, ..., 4.png. Noise_file_index argument points to the noise image, by default,
    it is set to 0. In case of generating single image this implementation might now look reasonable, however in case of generating whole dataset you don't have to care
    about anything. The code is just written this way that it's easier to use it for generating whole dataset in case of single images.

    :param epsilon: Epsilon value
    :type epsilon: float
    :param size: Size of the image (width, height), defaults to (640, 480)
    :type size: Tuple[int, int], optional
    :param ring_center: Position of central ring, defaults to (320, 240)
    :type ring_center: Tuple[int, int], optional
    :param brightness: Range of brightness, defaults to (80, 210)
    :type brightness: Tuple[int, int], optional
    :param noise_path: path to noise dataset, defaults to None
    :type noise_path: str, optional
    :param noise_file_index: Index (filename) of noise image used to generate image, optional
    :type noise_file_index: int, optional
    :return: 2D array which represents an image
    :rtype: np.array
    """
    _check_image(size, epsilon, ring_center, brightness, noise_file_index)
    
    
    pure_image = generate_pure_image(size, epsilon, ring_center, brightness)
    if noise_path:
        noise_image_filename = f"{noise_path}{noise_file_index}.png"
    else:
        noise_image_filename = pkg_resources.resource_filename(__name__, f"/samples/noise/{noise_file_index}.png")
    
    noise_image = cv2.imread(noise_image_filename)
    
    if ( noise_image.shape[:2] != size):
        noise_image = cv2.resize(noise_image, size, interpolation=cv2.INTER_AREA)
    noised_image = add_noise_to_image(pure_image, noise_image)
    return noised_image.astype(np.uint8)

if __name__ == "__main__":
    pass