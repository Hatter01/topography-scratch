import numpy as np
import cv2
import random as rd
import glob
import os

class DataGenerator:    
    def __init__(self, centered = True, noised = False, img_size = (640,480)) -> None:
        self.centered = centered
        self.noised = noised
        self.img_size = img_size
        
        ## adjust
        noise_data = np.array(glob.glob("data/noise/*.png"))
        self.noise = np.array([cv2.resize(cv2.imread(x, cv2.IMREAD_GRAYSCALE), img_size) for x in noise_data])

    def __apply_the_noise(self, img):
        noise = self.noise[np.random.randint(0,100)]
        prob1 = rd.randrange(6000, 8000) / 10000.
        x = img * prob1 + noise * (1 - prob1)
        return x


    def __generate_image(self, epsilon = 0, b_min = 100, b_max = 190, x_mid = 320, y_mid = 240):
        b_mean = b_max - b_min
        r_d = 640. * 640. / 6.
        img = np.fromfunction(lambda y, x:
            b_mean + (b_max - b_min) * np.cos(2 * np.pi * (epsilon + ((np.square(2*(x - x_mid)) + np.square(2*(y - y_mid))) / r_d)))
            ,(480, 640), dtype=np.float64).astype(np.uint8)

        if self.noised:
            img = self.__apply_the_noise(img)

        return img

    def __get_random_values(self):
        b_min = rd.randint(80, 120)
        b_max = rd.randint(170, 210)
        x_mid = rd.randint(270, 370)
        y_mid = rd.randint(190, 290)
        return b_min, b_max, x_mid, y_mid

    def __get_constant_values(self):
        b_min = rd.randint(80, 120)
        b_max = rd.randint(170, 210)
        return b_min, b_max, 320, 240

    def generate_images(self, dst: str, amount: int, clean_folder: bool = False):

        if clean_folder:
            for f in os.listdir(dst):
                os.remove(os.path.join(dst, f))

        values_generating_function = self.__get_constant_values
        if not self.centered:
            values_generating_function = self.__get_random_values
            
        for i in range(amount):
            epsilon = (i // (amount/1000))/1000.
            b_min, b_max, x_mid, y_mid = values_generating_function()

            ## change epsilon
            epsilon = np.round(rd.random(), decimals=3) 

            img = self.__generate_image(epsilon=epsilon, b_min=b_min, b_max=b_max, x_mid=x_mid, y_mid=y_mid)
            cv2.imwrite(dst + f'/{i}_{x_mid}_{y_mid}_{str(epsilon).replace(".","")}.png', img)
