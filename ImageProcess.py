#!/usr/bin/python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
from PIL import Image, ImageDraw
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

rootdir="./Images/"
img_path_list = []
img_num = 0
f_list = os.listdir(rootdir)
for i in f_list:
    if os.path.splitext(i)[1] == '.JPG':
        img_path_list.append(rootdir + i)
        img_num = img_num + 1
    img_path_list.sort()
print img_path_list
print img_num
if img_num > 0:
    I = plt.imread(img_path_list[0])
    plt.figure(1)
    plt.imshow(I)
    plt.hold('true')
    p1 = [0]*2
    p2 = [0]*2
    p3 = [0]*2
    p4 = [0]*2
    offset = [0]*2
    results_c = []
    results_r = []
    point = plt.ginput(2, timeout = 0)
    '''point1 = plt.ginput(1)
    point2 = plt.ginput(1)'''
    p1[0] = int(min(math.floor(point[0][0]), math.floor(point[1][0])))
    p1[1] = int(min(math.floor(point[0][1]), math.floor(point[1][1])))
    p2[0] = int(max(math.floor(point[0][0]), math.floor(point[1][0])))
    p2[1] = int(max(math.floor(point[0][1]), math.floor(point[1][1])))
    '''p1.append(math.floor(min(point[0][0],point[1][0])))
    p1.append(math.floor(min(point[0][1],point[1][1])))
    p2.append(math.floor(max(point[0][0],point[1][0])))
    p2.append(math.floor(max(point[0][1],point[1][1])))'''
    offset[0] = abs(math.floor(point[0][0]) - math.floor(point[1][0]))
    offset[1] = abs(math.floor(point[0][1]) - math.floor(point[1][1]))
    '''x = [p1[0], p1[0]+offset[0], p1[0]+offset[0], p1[0], p1[0]]
    y = [p1[1], p1[1], p1[1]+offset[1], p1[1]+offset[1], p1[1]]
    plt.plot(x,y,'r')'''
    II = I[p1[1]:p2[1],p1[0]:p2[0],:]
    plt.close()
    plt.figure(2)
    plt.imshow(II)
    plt.hold('true')
    point2 = plt.ginput(2, timeout = 0)
    p3[0] = int(min(math.floor(point2[0][0]), math.floor(point2[1][0])))
    p3[1] = int(min(math.floor(point2[0][1]), math.floor(point2[1][1])))
    p4[0] = int(max(math.floor(point2[0][0]), math.floor(point2[1][0])))
    p4[1] = int(max(math.floor(point2[0][1]), math.floor(point2[1][1])))
    img_g = rgb2gray(II)
    plt.close()
    for m in range(0, img_num):
        image = plt.imread(img_path_list[m])
        image_2 = image[p1[1]:p2[1],p1[0]:p2[0],:]
        Measure_image = rgb2gray(image_2)
        Measure_Matrix = Measure_image[p3[1]:p4[1], p3[0]:p4[0]]
        Result_Column = np.mean(Measure_Matrix, axis = 0)
        Result_Row = np.mean(Measure_Matrix, axis = 1)
        plt.figure(num = img_path_list[m])
        results_c.append(Result_Column)
        results_r.append(Result_Row)
        plt.subplot(2,2,1)
        plt.imshow(image)
        plt.subplot(2,2,2)
        plt.imshow(image_2)
        plt.subplot(2,2,3)
        plt.plot(Result_Column)
        plt.title('Vertical')
        plt.subplot(2,2,4)
        plt.plot(Result_Row)
        plt.title('Horizontal')
    plt.figure('Vertical')
    b = math.floor(math.sqrt(img_num));
    a = math.ceil(img_num / b);
    a = int(a)
    b = int(b)
    print a,b
    for j in range(0, img_num):
        plt.subplot(a,b,j+1)
        plt.plot(results_c[j])
    plt.figure('Horizontal')
    for k in range(0, img_num):
        plt.subplot(a,b,k+1)
        plt.plot(results_r[k])
    plt.show()
