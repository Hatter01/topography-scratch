import random as rd
from analyticalMethod import AnalyticalMathodNew, AnalyticalMathodOld
from src.data.dataGenerator import DataGenerator
import glob
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np



dg = DataGenerator(noised=True, centered=False)
am = AnalyticalMathodNew()

#x1 = time.time()
## adjust
dg.generate_images("data/processed/generated_data", 1, clean_folder=True)
#print(time.time()-x1)

## adjust
paths = glob.glob('data/processed/generated_data/*.png')
img = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)



vector = am.calculate_epsilon(img, check=True, blur_size=20, vector_filter_size=10)



