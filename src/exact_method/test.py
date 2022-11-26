import random as rd
from analyticalMethod import AnalyticalMathodNew, AnalyticalMathodOld
import glob
import cv2
import matplotlib.pyplot as plt
import time
import pandas as pd
from pathlib import Path, PurePath
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
parameters_path = project_dir.joinpath(PurePath("data/processed/parameters.csv")) 
parameters = pd.read_csv(parameters_path)

am = AnalyticalMathodNew()

mae = 0
i = 0
for index, image in parameters.iterrows():

    file_path = str(project_dir.joinpath(PurePath("data/processed/" + image["filename"])))
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # "x_y", "distances_from_x", "longer_sequence", "distances", "peaks_plot"
    eps = am.calculate_epsilon(img, check=["maxima", "x_y", "distances_from_x", "longer_sequence", "distances", "peaks_plot"], blur_size=10, vector_filter_size=20)
    print(eps)
    print()
    real_eps = image["epsilon"]
    real_x = image["ring_center_width"]
    real_y = image["ring_center_height"]
    x = np.abs(real_eps - eps)
    mae += min(x, 1 - x)
    print(real_eps, eps, min(x, 1 - x), real_x, real_y, "\n")
    i = index
    if min(x, 1 - x) > 0.1:
        break

print("MAE", mae/(i))






