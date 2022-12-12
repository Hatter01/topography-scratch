from typing import List, Tuple
import cv2
import numpy as np
import random
from tqdm import tqdm

import os
import pkg_resources

from generate_utils import save2directory, save2zip


def _check_args(num_images: int, num_used_raw_image: int):
    if num_images <= 0:
        raise ValueError("Number of generated images must be grater than 0.")
    if num_used_raw_image <= 20:
        raise ValueError(
            "Number of images used to extract noise must be greater than 20.")


def count_available_raw_images(path_to_raw: str) -> int:
    """Count how many raw images are available in given directory. If path_to_raw was not passed into the function, 
    package by default has 300 raw images installed itself, therefore function return 300 in case where that parameter is None.   


    :param path_to_raw: Path to folder which contains ONLY raw images. In that directory should not be any other files.
    :type path_to_raw: str
    :return: Number of different raw images which might be used for generating noise dataset.
    :rtype: int
    """
    if path_to_raw is None:
        return 300
    return len([name for name in os.listdir(path_to_raw) if os.path.isfile(os.path.join(path_to_raw, name))])


def generate_noise_image(size: Tuple[int, int] = (640, 480), num_used_raw_images: int = 50, path_to_raw: str = None, used_raw_images: np.array = None, seed: int = None) -> np.array:
    """Generate single noise image. The function is given indices (filenames) of raw images used for extraction. In case of generating single noise image (by this function)
    there is no need to parse argument with raw images filenames, the list of raw images will be automatically produced

    :param size: Size of generated noise image. Defaults to (640, 480).
    :type size: Tuple[int, int], optional
    :param num_used_raw_images: Number of raw images used to extract one noise, defaults to 100
    :type num_used_raw_images: int, optional
    :param path_to_raw: Path to raw images, defaults to None
    :type path_to_raw: str, optional
    :param used_raw_images: Indices of raw images used for extracting noise, defaults to None
    :type used_raw_images: np.array, optional
    :param seed: Set to some integer value to generate the same noise every function run, defaults to None
    :type seed: int, optional
    :return: Noise image
    :rtype: np.array
    """

    # this part is used only when we generate single noise image by this function and we don't use generate_noise_dataset function. TODO: For sure it's require review and optimization.


    if used_raw_images is None and path_to_raw is None:
        if seed is not None:
            np.random.seed(seed)
        used_raw_images = np.random.randint(0, 300, num_used_raw_images)
    if used_raw_images is None and path_to_raw is not None:
        if seed is not None:
            np.random.seed(seed)
        available_raw_images = count_available_raw_images(path_to_raw)
        used_raw_images = np.random.randint(
            0, available_raw_images, num_used_raw_images)

    raw_filenames = os.listdir(path_to_raw)


    noise_image = np.zeros(size[::-1])
    for _ in used_raw_images:
        # in case where path to raw images is not passed, read sample raw images from package (available only if you installed package via pip)
        if path_to_raw:
            raw_filename = path_to_raw + raw_filenames[_]
        else:
            raw_filename = pkg_resources.resource_filename(
                __name__, f"/samples/raw/{str(used_raw_images[_]).zfill(5)}.png")
        img = cv2.imread(raw_filename)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)[:, :, 0]
        noise_image = (noise_image+img)
    noise_image = noise_image/num_used_raw_images
    return noise_image.astype(np.uint8)


def generate_noise_dataset(path: str,
                           size: Tuple[int, int] = (640, 480),
                           num_images: int = 50,
                           num_used_raw_images: int = 100,
                           path_to_raw: str = None,
                           zipfile: bool = False,
                           zip_filename: str = None,
                           seed: int = None) -> None:
    """Create noise dataset. The way how seed work is as follows:
        1. We count how many raw images are available. If path_to_raw was not passed (available only when you installed package via pip)
            then this number is equal to 300 (package has 300 raw images installed, something like example datasets in scikit-learn).
            In case when path to raw was passed, then we count how many files are in that directory, therefore there should be only raw images and any other files.
            Moreover, raw images in that directory should be named like in the original dataset, always from 0-N with zeros at the beginning of the filename, so that
            in result filename contains always 5 chars. This is becuase for know the code assumes that for example if directory contains 200 images, then we have there
            200 images named from 00000.png, 00001.png, ..., 00199.png. IT WON'T WORK IN CASE OF ANY OTHER FILES ORGANIZATION.
        2. Numpy generates random list of images indices which will be used to extract noise, if you use seed, this list will be same all the time.

    :param path: Path where images should be stored
    :type path: str
    :param size: Size of generated noise image. Defaults to (640, 480).
    :type size: Tuple[int, int], optional
    :param num_images: Number of images that will be created, defaults to 50
    :type num_images: int, optional
    :param num_used_raw_images: Raw images used to extract one noise, defaults to 100
    :type num_used_raw_images: int, optional
    :param path_to_raw: Path to raw images, defaults to None
    :type path_to_raw: str, optional
    :param zipfile: Set to True if you want to save noise in zipfile, defaults to False
    :type zipfile: bool, optional
    :param zip_filename: Name of the zipfile, should be passed only if zipfile=True, defaults to None
    :type zip_filename: str, optional
    :param seed: Set seed to obtain the same result, defaults to None
    :type seed: int, optional
    """
    _check_args(num_images, num_used_raw_images)

    raw_images_needed = num_images*num_used_raw_images
    available_raw_images = count_available_raw_images(path_to_raw)

    if seed is not None:
        np.random.seed(seed)
    # all raw images selected for noise extraction
    selected_raw_images = np.random.randint(
        0, available_raw_images, size=raw_images_needed)

    # raw images per one noise image
    raw_images_per_frame = np.split(selected_raw_images, num_images)

    for frame in tqdm(range(num_images)):
        noise_image = generate_noise_image(
            size, num_used_raw_images, path_to_raw, raw_images_per_frame[frame])
        if zipfile:
            save2zip(
                noise_image, img_filename=f"{frame}.png", filename=zip_filename, path=path)
        else:
            save2directory(noise_image, img_filename=f"{frame}.png", path=path)


if __name__ == "__main__":
    pass
