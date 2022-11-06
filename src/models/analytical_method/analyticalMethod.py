import numpy as np
import cv2 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class AnalyticalMathod:
    def __init__(self) -> None:
        pass

    def draw_circle(self, img, x, y):
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
        return img

    def detect_middle_x_y(self, img):
        raise NotImplementedError("Must override detect_middle")

    def detect_middle_x(self, img, filter_size = 10, check = False):
        vector = img.mean(axis=0)
        if check:
            plt.plot(vector)

        vector = cv2.filter2D(vector, -1, np.ones(filter_size)/ filter_size).flatten()
        if check:
            plt.figure()
            plt.plot(vector)
            plt.savefig("vector.png")

        # find maxima of the function 
        maxima = find_peaks(vector, distance=5)[0]
        # distacne between each consecutive maximum
        distances = np.array([-maxima[i] + maxima[i+1] for i in range(len(maxima)-1)])

        # select top 2 biggest distances
        x1, x2 = (-distances).argsort()[:2]
        small_x = np.min([x1, x2])
        big_x = np.max([x1, x2])

        if np.fabs(distances[small_x + 1]-distances[big_x - 1]) == 0 or np.fabs(distances[small_x-1]-distances[big_x+1]) < 5:
            return maxima, distances, (maxima[small_x] + maxima[big_x+1]) // 2
        else:
            return maxima, distances, (maxima[x1 - 1] + maxima[x1 + 2]) // 2

    def middle_idx(self, vector):
        i1, i2 = 0, len(vector) - 1
        while np.fabs(vector[i1] - vector[i2]) > 2:
            if vector[i1] > vector[i2]:
                i1 += 1
            else:
                i2 -= 1
        return int(np.ceil((i2 + i1) / 2 ))

    def calculate_epsilon(self, img, check = False, blur_size = 20, vector_filter_size = 5):

        # blur to reduce noise
        img_tmp = cv2.filter2D(img, -1, np.ones(blur_size**2).reshape((blur_size,blur_size))/blur_size**2)

        # coordinates of center
        x, y = self.detect_middle_x_y(img_tmp)

        # middle strip
        strip = img_tmp[y-20 : y+20, :]
        if check:
            cv2.imwrite("strip.png", strip)

        # find middle more accurately
        maxima, distances, x_new = self.detect_middle_x(strip, vector_filter_size, check= True)
        distances_from_x = np.fabs(maxima - x_new)

        if check:
            print("MAXIMA")
            print(distances)
            print(maxima)
            print(distances_from_x)
            print(x_new, y)
        
        idx = self.middle_idx(distances_from_x)
        left_distances, right_distances = distances_from_x[:idx][::-1], distances_from_x[idx:]
        longer_sequence = left_distances
        if len(right_distances)>len(left_distances): longer_sequence = right_distances
        print(longer_sequence)
        
        if check:
            plt.figure()
            plt.plot(np.arange(1, len(longer_sequence)+1), (longer_sequence)**2)
            plt.savefig("plot.png")
        

        return None


        


class AnalyticalMathodNew(AnalyticalMathod):
    def detect_middle_x_y(self, img):
        # https://theailearner.com/tag/cv2-houghcircles/
        detected_circles = cv2.HoughCircles(
            img, cv2.HOUGH_GRADIENT, 1, 10, param1 = 50, param2 = 30, minRadius = 250, maxRadius = 300
        )

        # check whether there are circles detected
        if detected_circles is None:
            return None

        # round results
        detected_circles = np.uint16(np.around(detected_circles))  

        # taking the first (most accurate) coordinates
        most_accurate = detected_circles[0, 0]
        x, y, r = most_accurate

        return x, y

class AnalyticalMathodOld(AnalyticalMathod):
           

    def detect_middle_x_y(self, img):
        img_strip_x =  img[240-40 : 240+40, :]
        _,_,x = self.detect_middle_x(img=img_strip_x)
        
        img_strip_y = np.transpose(img[:, x - 20: x + 20])
        _,_,y = self.detect_middle_x(img=img_strip_y)

        img_strip_x =  img[y-20 : y+20, :]
        _,_,x = self.detect_middle_x(img=img_strip_x)
        return x, y

        



