import random as rd
from analyticalMethod import AnalyticalMathodNew, AnalyticalMathodOld
import glob
import cv2
import matplotlib.pyplot as plt
import time
import pandas as pd
from pathlib import Path, PurePath
import numpy as np
from tqdm import tqdm

project_dir = Path(__file__).resolve().parents[2]
parameters_path = project_dir.joinpath(PurePath("data/processed/parameters.csv")) 
parameters = pd.read_csv(parameters_path)

old = AnalyticalMathodOld()
new = AnalyticalMathodNew()

mae_old_method = 0
mae_new_method = 0
time_old_method = 0
time_new_method = 0
print("TESTING:")
for index, image in tqdm(parameters.iterrows()):

    file_path = str(project_dir.joinpath(PurePath("data/processed/" + image["filename"])))
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    real_eps = image["epsilon"]
    real_x = image["ring_center_width"]
    real_y = image["ring_center_height"]

    # possible parameters to check
    # "x_y", "distances_from_x", "longer_sequence", "distances", "peaks_plot", "plot"

    time_old_method -= time.time() 
    eps_old = old.calculate_epsilon(img, check=[], blur_size=10, vector_filter_size=20)
    time_old_method += time.time() 
    x_old = np.abs(real_eps - eps_old)
    mae_old_method += min(x_old, 1 - x_old)

    time_new_method -= time.time() 
    eps_new = new.calculate_epsilon(img, check=[], blur_size=20, vector_filter_size=20)
    time_new_method += time.time() 
    x_new = np.abs(real_eps - eps_new)
    mae_new_method += min(x_new, 1 - x_new)

print()
print("OLD METHOD:")
print("MAE: ", mae_old_method/len(parameters))
print("IMAGES PER SECOND: ", len(parameters)/time_old_method)
print()
print("NEW METHOD:")
print("MAE: ", mae_new_method/len(parameters))
print("IMAGES PER SECOND: ", len(parameters)/time_new_method)






