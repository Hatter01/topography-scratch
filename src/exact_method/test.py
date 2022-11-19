import random as rd
from analyticalMethod import AnalyticalMathodNew, AnalyticalMathodOld
import glob
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np



am = AnalyticalMathodOld()

paths = glob.glob('data/processed/*.png')
paths = rd.sample(paths, 1)

mae = 0
for path in paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # "x_y", "distances_from_x", "longer_sequence", "distances", "peaks_plot"
    eps = am.calculate_epsilon(img, check=[], blur_size=15, vector_filter_size=20)
    print(eps)
    print()
#     real_eps = int(path.split(".")[0].split("_")[-1][1:])/1000
#     real_x = int(path.split(".")[0].split("_")[-3])
#     real_y = int(path.split(".")[0].split("_")[-2])
#     x = np.abs(real_eps - eps)
#     mae += min(x, 1 - x)
#     print(path, real_eps, eps, min(x, 1 - x), real_x, real_y, "\n")

# print("MAE", mae/N)






