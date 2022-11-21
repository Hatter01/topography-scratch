from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os


from src.data.generate_image import generate_image
from src.data._models import ImageDetails

from src.data.generate_utils import save2directory, save2zip



def _check_args(path: str, n_copies: int, epsilon_step: float, zipfile: bool, filename: str) -> None:
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Provided path: {path} is not directory.")
    if n_copies <= 0:
        raise ValueError("Number of created copies must be positive.")
    if epsilon_step < 0.0 or epsilon_step > 1.0:
        raise ValueError("Epsilon step must be in range <0.0 - 1.0>.")
    if zipfile:
        if not filename:
            raise Exception("In case of exporting dataset to zipfile, filename must be provided.")
        if filename[-4:] != ".zip":
            raise Exception("Provide filename with .zip extension, e.g. \"dataset.zip\"")        
        

def parameters2csv(parameters: List[Dict], path: str, parameters_filename: str) -> None:
    """Save parameters to .csv file

    :param parameters: Parameters for each image
    :type parameters: List[Dict]
    :param path: Path to the output directory
    :type path: str
    :param parameters_filename: Name of the parameters file
    :type parameters_filename: str
    """
    df = pd.DataFrame.from_dict(parameters)
    df.to_csv(path+parameters_filename, encoding='utf-8', index=False)
    
def count_available_noises(noise_path: str) -> int:
    if noise_path is None:
        return 25
    return len([name for name in os.listdir(noise_path) if os.path.isfile(os.path.join(noise_path, name))])

# TODO: correct docstrings
def generate_balanced_dataset(path: str,
                              n_copies: int,
                              epsilon_range: Tuple[float, float]=(0.0, 1.0), 
                              epsilon_step: float=0.001,
                              size: Tuple[int, int]=(640, 480),
                              brightness: Tuple[int, int]=(80,210),
                              center_shift: float = 0.01,
                              zipfile: bool=False,
                              filename: str=None,
                              save_parameters: bool=True,
                              parameters_filename: str="parameters.csv",
                              noise_path: str=None,
                              seed: int = None
                              ) -> None:
    """Generate balanced dataset and save to the output directory or .zip file. Noise_path argument need to be passed if you didn't install package via pip.

    :param path: Path where output images or compressed .zip file should be stored
    :type path: str
    :param n_copies: Number of images that has to be created with the same epsilon value.
    :type n_copies: int
    :param epsilon_range: Range of epsilons values used to generate images, defaults to (0.0, 1.0)
    :type epsilon_range: Tuple[float, float], optional
    :param epsilon_step: Step by epsilon value increases every iteration, defaults to 0.001
    :type epsilon_step: float, optional
    :param size: Size of generated images (width, height), defaults to (640, 480)
    :type size: Tuple[int, int], optional
    :param brightness: Brightness range of each pixel, defaults to (80,210)
    :type brightness: Tuple[int, int], optional
    :param center_shift: Percentage of random shifting ring center in the image, defaults to 0.01.
    :type brightness: Tuple[int, int], optional
    :param zipfile: Set to True if output images should be compressed to .zip file, defaults to False
    :type zipfile: bool, optional
    :param filename: Name of output .zip file. Need to be provided if zipfile is True, defaults to None
    :type filename: str, optional
    :param save_parameters: Set to False if additional file with each image parameters should not be stored, defaults to True
    :type save_parameters: bool, optional
    :param parameters_filename: Name of parameters file, defaults to "parameters.csv"
    :type parameters_filename: str, optional
    :param noise_path: Path to the noise dataset, optional
    :type noise_path: str, optional
    :param seed: Set seed to create identical dataset each time.
    :type seed: int, optional
    """
    _check_args(path, n_copies, epsilon_step, zipfile, filename)
    
    if seed:
        random.seed(seed)
    
    min_epsilon, max_epsilon = epsilon_range
    width, height = size
    
    max_width_center_shift =  width * center_shift
    min_width_center = int(width/2 - max_width_center_shift)
    max_width_center = int(width/2 + max_width_center_shift)
    
    max_height_center_shift = height * center_shift
    min_height_center = int(height/2 - max_height_center_shift)
    max_height_center = int(height/2 + max_height_center_shift)
    

    
    img_index = 0
    parameters: List[Dict] = []
    epsilons = np.arange(start=min_epsilon, stop=max_epsilon, step=epsilon_step)
    
    available_noises = count_available_noises(noise_path)
    
    # create arrays with ring_center position and choosen noises. Those arrays will be always the same if you set the seed. 
    if seed is not None:
        np.random.seed(seed)
    width_centers = np.random.randint(min_width_center, max_width_center+1, len(epsilons)*n_copies)
    height_centers = np.random.randint(min_height_center, max_height_center+1, len(epsilons)*n_copies)
    choosen_noises = np.random.randint(0, available_noises, len(epsilons)*n_copies)
    
    for _epsilon in tqdm(epsilons):
        _epsilon = float("{:.3f}".format(_epsilon))
        for _ in range(n_copies):
            ring_center = (width_centers[img_index],
                           height_centers[img_index])
            
            img = generate_image(_epsilon, size, ring_center, brightness, noise_path, choosen_noises[img_index])
            img_filename = f"{str(img_index).zfill(5)}.png"
            
            if zipfile:
                save2zip(img, img_filename, filename, path)
            else:
                save2directory(img, img_filename ,path)
            
            if save_parameters:
                img_details = ImageDetails( filename=img_filename,
                                               width=width,
                                                height=height,
                                                epsilon=_epsilon,
                                                ring_center_width=ring_center[0],
                                                ring_center_height=ring_center[1],
                                                min_brightness=brightness[0],
                                                max_brightness=brightness[1],
                                                used_noise=choosen_noises[img_index]
                                                )
                parameters.append(img_details.dict())
            img_index += 1
    parameters2csv(parameters, path, parameters_filename)


if __name__ == "__main__":
    pass