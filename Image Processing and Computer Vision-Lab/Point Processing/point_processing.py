import sys
sys.path.append("C:\\Users\\SSD\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def Gamma(img,gamma):
    L=img.shape[0]-1
    h=img.shape[0]
    w=img.shape[1]
    #result=np.empty_like(img)
    result=np.zeros((img.shape[0],img.shape[1]))
    for i in range(h):
        for j in range(w):
            result[i, j] = ((img[i, j] / L) ** gamma) * L
    res = np.round(result).astype(np.uint8)
    cv.imshow('Gamma Output',res)





def inverse_log(img):
    c = 255/ np.log(1 + np.max(img))
    h=img.shape[0]
    w=img.shape[1]
    s=np.zeros((h,w)) 
    r=img
    s = c *np.log2(1+r)
   
    inv_s = np.power(2, s / c) - 1
    
    inv_s=np.round(inv_s).astype(np.uint8)
    cv.imshow('inverse_log',inv_s)

def contrast_stretching(img):
    h=img.shape[0]
    w=img.shape[1]
    maxiI = np.max(img)
    miniI = np.amin(img)
    maxoI = 511 #127 
    minoI = 256 #0
    contrast = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            r=img[i,j]
            contrast[i,j] = (maxoI-minoI)*(r-miniI)/(maxiI-miniI)
    contrast=np.round(contrast).astype(np.uint8)
    cv.imshow('contrast output',contrast)



img = cv.imread('./lena.png',0)
print(img.shape)
cv.imshow('input',img)

Gamma(img, 1/2.0)
#Gamma(img, 2.0)
inverse_log(img)
contrast_stretching(img)


cv.waitKey(0)
cv.destroyAllWindows()

