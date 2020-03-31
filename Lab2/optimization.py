import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
import copy
import math


class optimizeImage:
    def cleanOverX(self, imageMatrix):
        subset = imageMatrix
        row, col = np.shape(subset)
        # cleaning the photo [x]
        for i in range(row - 1):
            for j in range(col - 1):

                if i == 0 & j == 0:
                    cnt = 3
                    if subset[i + 1][j] != subset[i][j]: cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]: cnt -= 1
                    if subset[i][j + 1] != subset[i][j]: cnt -= 1
                    if cnt == 1:
                        if subset[i][j] == 0 : subset[i][j] = 1
                        else: subset[i][j] = 0

                elif i == 0 & j == row:
                    cnt = 3
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                elif i == row & j == 0:
                    cnt = 3
                    if subset[i - 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                elif i == row & j == row:
                    cnt = 3
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                elif i == 0:
                    cnt = 5

                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                elif i == row:
                    cnt = 5

                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                elif j == 0:
                    cnt = 5

                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                elif j == row:
                    cnt = 5

                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                else:
                    cnt = 9

                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1
                    if cnt <= 3:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0
        return subset

    def cleanOverY(self, imageMatrix):
        subset = imageMatrix
        row, col = np.shape(subset)
        # cleaning the photo [y]
        for j in range(col - 1):
            for i in range(row - 1):

                if i == 0 & j == 0:

                    cnt = 3

                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                elif i == 0 & j == row:
                    cnt = 3
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                elif i == row & j == 0:
                    cnt = 3
                    if subset[i - 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                elif i == row & j == row:
                    cnt = 3
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                elif i == 0:
                    cnt = 5

                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                elif i == row:
                    cnt = 5

                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                elif j == 0:
                    cnt = 5

                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                elif j == row:
                    cnt = 5

                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if cnt == 1:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0

                else:
                    cnt = 9

                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1
                    if cnt <= 3:

                        if subset[i][j] == 0:
                            subset[i][j] = 1
                        else:
                            subset[i][j] = 0
        return subset

    def holeIsland(self, imageMatrix):
        subset = imageMatrix
        row, col = np.shape(subset)

        # fill holes [1/4]
        for i in range(round(row / 2))[::-1]:
            for j in range(round(col / 2))[::-1]:

                cnt = 8
                if subset[i][j] == 1:
                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:
                        subset[i][j] = 0
        # eliminate islands [1/4]
        for i in range(round(row / 2)):
            for j in range(round(col / 2)):

                cnt = 8
                if subset[i][j] == 0:
                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:
                        subset[i][j] = 1

        # fill holes [2/4]
        for i in range(round(row / 2)):
            for j in range(round(col / 2), col - 1)[::-1]:

                cnt = 8
                if subset[i][j] == 0:
                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:
                        subset[i][j] = 1
        # eliminate islands [2/4]
        for i in range(round(row / 2))[::-1]:
            for j in range(round(col / 2), col - 1):

                cnt = 8
                if subset[i][j] == 1:
                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:
                        subset[i][j] = 0

        # fill holes [3/4]
        for i in range(round(row / 2), row - 1)[::-1]:
            for j in range(round(col / 2)):

                cnt = 8
                if subset[i][j] == 0:
                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:
                        subset[i][j] = 1
        # eliminate islands [3/4]
        for i in range(round(row / 2), row - 1):
            for j in range(round(col / 2))[::-1]:

                cnt = 8
                if subset[i][j] == 1:
                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:
                        subset[i][j] = 0

        # fill holes [4/4]
        for i in range(round(row / 2), row - 1)[::-1]:
            for j in range(round(col / 2), col - 1)[::-1]:

                cnt = 8
                if subset[i][j] == 0:
                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:
                        subset[i][j] = 1
        # eliminate islands[4/4]
        for i in range(round(row / 2), row - 1):
            for j in range(round(col / 2), col - 1):

                cnt = 8
                if subset[i][j] == 1:
                    if subset[i][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i + 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j + 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i - 1][j - 1] != subset[i][j]:
                        cnt -= 1
                    if subset[i][j - 1] != subset[i][j]:
                        cnt -= 1

                    if cnt <= 4:
                        subset[i][j] = 0

        return subset

    def cleanBorder(self, imageMatrix):
        subset = imageMatrix
        row, col = np.shape(subset)
        for i in range(row - 1):
            subset[i][0] = 0
        for i in range(col - 1):
            subset[0][i] = 0
        for i in range(row - 1):
            subset[i][col - 1] = 0
        for i in range(col - 1):
            subset[row - 1][i] = 0
        return subset

    def plot_im(self, jpg, title=None, save=0, filename=None):
        self.jpg = jpg
        self.title = title
        plt.figure()
        plt.imshow(self.jpg)
        plt.title(self.title)
        imageDir = "Images/"
        if not os.path.exists(imageDir):
            os.makedirs(imageDir)
        titleToSave = title.replace(' ', '_').replace(':', '')
        plt.savefig(imageDir + filename + "_" + titleToSave + ".png")
        if save != 0:
            plt.savefig('')
        plt.show()
        plt.pause(0.1)

    def plot_math(self, matrix, title=None, save=0, filename=None):
        plt.matshow(matrix)
        plt.title(title)
        imageDir = "Images/"
        if not os.path.exists(imageDir):
            os.makedirs(imageDir)
        titleToSave = title.replace(' ', '_').replace(':', '')
        plt.savefig(imageDir + filename + "_" + titleToSave + ".png")
        if save != 0:
            plt.savefig('')
        plt.show()
