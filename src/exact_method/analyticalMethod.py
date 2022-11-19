import numpy as np
import cv2 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression



class AnalyticalMathod:
    def __init__(self) -> None:
        pass

    def draw_circle(self, img, x, y):
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
        return img

    def detect_middle_x_y(self, img):
        raise NotImplementedError("Must override detect_middle")

    def detect_middle_x(self, img, filter_size = 10, check = [], mode="x"):
        if "strip" in check:
            cv2.imwrite("strip.png", img=img)
        vector = img.mean(axis=0)

        vector = cv2.filter2D(vector, -1, np.ones(filter_size)/ filter_size).flatten()
        if "peaks_plot" in check:
            plt.figure()
            plt.plot(vector)
            plt.savefig("vector.png")

        # find maxima of the function 
        maxima = find_peaks(vector, distance=5)[0]
        if mode == "y":
            minima = find_peaks(-vector, distance=5)[0]
            if len(minima) > len(maxima):
                maxima = minima

        if "maxima" in check:
            print("Maxima")
            print(maxima)

        # distacne between each consecutive maximum
        distances = np.array([-maxima[i] + maxima[i+1] for i in range(len(maxima)-1)])
        if "distances" in check:
            print("Distances")
            print(distances)

        # select top 2 biggest distances
        x1, x2 = (-distances).argsort()[:2]
        small_x = np.min([x1, x2])
        big_x = np.max([x1, x2])
        
        ## need change
        # if small_x == 0:
        #     return maxima, distances, (maxima[small_x + 1])
        # if big_x == len(distances)-1 :
        #     return maxima, distances, (maxima[big_x - 1])
        if np.fabs(distances[small_x + 1]-distances[big_x - 1]) == 0 or np.fabs(distances[small_x-1]-distances[big_x+1]) < 5:
            return maxima, distances, (maxima[small_x] + maxima[big_x+1]) // 2
        else:
            return maxima, distances, (maxima[x1 - 1] + maxima[x1 + 2]) // 2

    def middle_idx(self, vector):
        i1, i2 = 0, len(vector) - 1
        while np.fabs(vector[i1] - vector[i2]) >= 2:
            if vector[i1] > vector[i2]:
                i1 += 1
            else:
                i2 -= 1
        return i1, i2

    def calculate_epsilon(self, img, check = [], blur_size = 20, vector_filter_size = 10):
        if check:
            print("ANALYSIS")
        # blur to reduce noise
        img_tmp = cv2.filter2D(img, -1, np.ones(blur_size**2).reshape((blur_size,blur_size))/blur_size**2)

        # coordinates of center
        x, y = self.detect_middle_x_y(img_tmp, vector_filter_size)

        # middle strip
        strip = img_tmp[y-20 : y+20, :]

        # find middle more accurately
        maxima, _, x_new = self.detect_middle_x(strip, vector_filter_size, check= check)
        distances_from_x = np.fabs(maxima - x_new)

        if "distances_from_x" in check:
            print("Distances from the middle")
            print(distances_from_x)

        if "x_y" in check:
            print("X   Y")
            print(x_new,y)
        
        idx1, idx2 = self.middle_idx(distances_from_x)
        idx = (idx1+idx2)//2
        if (idx2 - idx1) % 2 ==0:
            left_distances, right_distances = distances_from_x[:idx][::-1], distances_from_x[idx + 1:]
        else:
            left_distances, right_distances = distances_from_x[:idx + 1][::-1], distances_from_x[idx + 1:]

        longer_sequence = left_distances
        if len(right_distances)>len(left_distances): longer_sequence = right_distances
        
        if "longer_sequence" in check:
            print("The longer part")
            print(longer_sequence)

        if longer_sequence[1] < longer_sequence[2] - longer_sequence[1]:
            longer_sequence = longer_sequence[2:]
        if longer_sequence[0] < longer_sequence[1] - longer_sequence[0]:
            longer_sequence = longer_sequence[1:]

        
        if "longer_sequence" in check:
            print(longer_sequence)

        if "plot" in check:
            plt.figure()
            plt.plot(np.arange(1, len(longer_sequence)+1), (longer_sequence)**2)
            plt.savefig("plot.png")

        x = np.arange(1, len(longer_sequence) + 1)
        x = np.expand_dims(x, axis=1)
        #longer_sequence =np.array([ 25, 130, 184, 226, 261, 292])
        y = longer_sequence ** 2
        regressor = LinearRegression().fit(x,y)

        intercept = regressor.predict([[0]])
        slope = regressor.coef_
        epsilon = np.fabs(1+intercept/slope) % 1.
        

        return epsilon


        


class AnalyticalMathodNew(AnalyticalMathod):
    def detect_middle_x_y(self, img, vector_filter_size = None):
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
           

    def detect_middle_x_y(self, img, vector_filter_size):
        img_strip_x =  img[240-40 : 240+40, :]
        _,_,x = self.detect_middle_x(img=img_strip_x, filter_size=vector_filter_size)
        
        img_strip_y = np.transpose(img[:, x - 20: x + 20])
        _,_,y = self.detect_middle_x(img=img_strip_y, 
            filter_size=vector_filter_size,
            check=["x_y", "distances", "maxima", "distances_from_x", "longer_sequence", "peaks_plot", "strip"])

        img_strip_x =  img[y-20 : y+20, :]
        _,_,x = self.detect_middle_x(img=img_strip_x, filter_size=vector_filter_size)
        return x, y

        


