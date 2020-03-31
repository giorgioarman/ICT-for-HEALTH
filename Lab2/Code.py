import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import copy
import math
import Lab2.optimization as opti
import os


opt = opti.optimizeImage()
count = 0

np.set_printoptions(precision=2)  # use only two decimal digits when printing numbers
plt.close('all')  # close previously opened pictures

for item in os.listdir("moles/"):
    fileName = item.split('.')[0]

    file = './moles/' + item
    im_or = mpimg.imread(file)
    # im_or is Ndarray 583 x 584 x 3 unint8
    # plot the image, to check it is correct:

    opt.plot_im(im_or, title='original image', save=1, filename=fileName)

    # %% reshape the image from 3D to 2D
    N1, N2, N3 = im_or.shape  # note: N3 is 3, the number of elementary colors, i.e. red, green ,blue
    # im_or(i,j,1) stores the amount of red for the pixel in position i,j
    # im_or(i,j,2) stores the amount of green for the pixel in position i,j
    # im_or(i,j,3) stores the amount of blue for the pixel in position i,j
    # we resize the original image
    im_2D = im_or.reshape((N1 * N2, N3))  # im_2D has N1*N2 row and N3 columns
    # pixel in position i.j goes to position k=(i-1)*N2+j)
    # im_2D(k,1) stores the amount of red of pixel k
    # im_2D(k,2) stores the amount of green of pixel k
    # im_2D(k,3) stores the amount of blue of pixel k
    # im_2D is a sequence of colors, that can take 2^24 different values
    Nr, Nc = im_2D.shape
    # %% get a simplified image with only Ncluster colors
    # number of clusters/quantized colors we want to have in the simpified image:
    Ncluster = 3
    # instantiate the object K-means:
    kmeans = KMeans(n_clusters=Ncluster, random_state=0)
    # run K-means:
    kmeans.fit(im_2D)
    # get the centroids (i.e. the 3 colors). Note that the centroids
    # take real values, we must convert these values to uint8
    # to properly see the quantized image
    kmeans_centroids = kmeans.cluster_centers_.astype('uint8')
    # copy im_2D into im_2D_quant
    im_2D_quant = im_2D.copy()
    for kc in range(Ncluster):
        quant_color_kc = kmeans_centroids[kc, :]
        # kmeans.labels_ stores the cluster index for each of the Nr pixels
        # find the indexes of the pixels that belong to cluster kc
        ind = (kmeans.labels_ == kc)
        # set the quantized color to these pixels
        im_2D_quant[ind, :] = quant_color_kc
    im_quant = im_2D_quant.reshape((N1, N2, N3))
    opt.plot_im(im_quant, title='image with quantized colors', save=1, filename=fileName)

    # %% Find the centroid of the main mole

    # %% Preliminary steps to find the contour after the clustering
    #
    # 1: find the darkest color found by k-means, since the darkest color
    # corresponds to the mole:
    centroids = kmeans_centroids
    sc = np.sum(centroids, axis=1)
    i_col = sc.argmin()  # index of the cluster that corresponds to the darkest color
    # 2: define the 2D-array where in position i,j you have the number of
    # the cluster pixel i,j belongs to
    im_clust = kmeans.labels_.reshape(N1, N2)
    # plt.matshow(im_clust)
    # 3: find the positions i,j where im_clust is equal to i_col
    # the 2D Ndarray zpos stores the coordinates i,j only of the pixels
    # in cluster i_col
    zpos = np.argwhere(im_clust == i_col)
    # 4: ask the user to write the number of objects belonging to
    # cluster i_col in the image with quantized colors

    N_spots_str = input("How many distinct dark spots can you see in the image? ")
    plt.close('all')
    N_spots = int(N_spots_str)

    # 5: find the center of the mole
    if N_spots == 1:
        center_mole = np.median(zpos, axis=0).astype(int)
    else:
        # use K-means to get the N_spots clusters of zpos
        kmeans2 = KMeans(n_clusters=N_spots, random_state=0)
        kmeans2.fit(zpos)
        centers = kmeans2.cluster_centers_.astype(int)
        # the mole is in the middle of the picture:
        center_image = np.array([N1 // 2, N2 // 2])
        center_image.shape = (1, 2)
        d = np.zeros((N_spots, 1))
        for k in range(N_spots):
            d[k] = np.linalg.norm(center_image - centers[k, :])
        center_mole = centers[d.argmin(), :]

    # 6: take a subset of the image that includes the mole
    c0 = center_mole[0]
    c1 = center_mole[1]
    RR, CC = im_clust.shape
    stepmax = min([c0, RR - c0, c1, CC - c1])
    cond = True
    area_old = 0
    surf_old = 1
    step = 10  # each time the algorithm increases the area by 2*step pixels
    # horizontally and vertically
    im_sel = (im_clust == i_col)  # im_sel is a boolean NDarray with N1 row and N2 columns
    im_sel = im_sel * 1  # im_sel is now an integer NDarray with N1 row and N2 columns
    while cond:
        subset = im_sel[c0 - step:c0 + step + 1, c1 - step:c1 + step + 1]
        area = np.sum(subset)
        Delta = np.size(subset) - surf_old
        surf_old = np.size(subset)
        if area > area_old + 0.01 * Delta:
            step = step + 10
            area_old = area
            cond = True
            if step > stepmax:
                cond = False
        else:
            cond = False

    # subset is the search area

    opt.plot_math(subset, title='subset', save=1, filename=fileName)

    row, col = np.shape(subset)

    # cleaning the photo [x]
    subset = opt.cleanOverX(subset)
    # cleaning the photo [y]
    subset = opt.cleanOverY(subset)

    # Fill the holes and eliminate islands
    subset = opt.holeIsland(subset)

    # clean borders
    subset = opt.cleanBorder(subset)

    # final Checking the image and Fill the holes and eliminate islands
    subset = opt.cleanOverX(subset)
    subset = opt.cleanOverY(subset)
    if row >= 150:
        subset = opt.holeIsland(subset)

    opt.plot_math(subset, title='Cleaned subset', save=1, filename=fileName)

    # perimeter and area
    perimeter = np.copy(subset)

    for i in range(row - 1):
        for j in range(col - 1):
            if subset[i][j] == 1 and subset[i - 1][j] == 1 and subset[i + 1][j] == 1 and subset[i][j - 1] == 1 and \
                    subset[i][j + 1] == 1:
                perimeter[i][j] = 0

    Perimeter_mole = 0
    for i in range(row - 1):
        for j in range(col - 1):
            if perimeter[i][j] == 1:
                Perimeter_mole += 1

    opt.plot_math(perimeter, title='perimeter', save=1, filename=fileName)

    Area_mole = 0
    for i in range(row - 1):
        for j in range(col - 1):
            if subset[i][j] == 1:
                Area_mole += 1

    # Calculations
    Perimeter_circle = 2 * (math.sqrt(math.pi)) * (math.sqrt(Area_mole))

    ratio = round(Perimeter_mole / Perimeter_circle, 2)
    print(ratio)

    # SAVING THE RESULT IN A FILE
    with open('ratio.txt', "a") as out_file:
        out_file.write(str(fileName + ",") + str(ratio))
        out_file.write('\n')

