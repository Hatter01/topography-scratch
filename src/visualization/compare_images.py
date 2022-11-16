from typing import Tuple
import cv2
import numpy as np


def compute_mse(img1: np.array, img2: np.array) -> Tuple[float, np.array]:
    
    assert img1.shape == img2.shape
    
    width, height = img1.shape[0], img1.shape[1]
    
    difference = cv2.subtract(img1, img2)
    error = np.sum(difference**2)
    mse = error/float(width*height)
    return mse, difference



if __name__ == "__main__":
    pass
    