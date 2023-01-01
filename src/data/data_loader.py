import numpy as np
import cv2
import os
import pandas as pd
from pathlib import Path, PurePath

#stransformation on the image (resizing)
def transform_image(path_images,file_name):
    img = cv2.imread(path_images + file_name, cv2.IMREAD_GRAYSCALE)
    return img

def load_dataset():

    project_dir = Path(__file__).resolve().parents[2]
    processed = str(project_dir.joinpath(PurePath("data/processed"))) + os.sep

    path_images = processed
    dataset = pd.read_csv(processed + "parameters.csv")

    y = np.array(dataset['epsilon'])
    X = np.array(list(map(lambda file_name:  transform_image(path_images,file_name) , dataset["filename"])))

    print("X.shape", X.shape)
    return (X,y)

load_dataset()
