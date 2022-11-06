import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

## adjust
real_data = np.array(glob.glob("data/real_data/1channel/*.png"))
shuffled = np.random.shuffle(real_data)
splitted = np.array_split(real_data, 100)

for idx, chunk in enumerate(splitted):
    images = np.array([cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in chunk])
    noise_image = np.average(images, axis=0)
    cv2.imwrite(f'data/noise/{idx}.png', noise_image)