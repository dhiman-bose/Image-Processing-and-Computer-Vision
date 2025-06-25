# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 00:16:01 2023

@author: SSD
"""

import sys
sys.path.append("C:\\Users\\SSD\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")

import cv2 as cv


import numpy as np


def add_gaussian_noise(image, sigma1,sigma2):
    noise=np.zeros((image.shape[0],image.shape[1]),dtype=np.float32)
    midx=0
    midy=0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x=i-midx
            xx=x*x
            y=j-midy
            yy=y*y
            power=-(xx+yy)/(2*sigma1*sigma1)
            noise[i,j]=np.exp(power)
    midx=image.shape[0]-1
    midy=image.shape[1]-1
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x=i-midx
            xx=x*x
            y=j-midy
            yy=y*y
            power=-(xx+yy)/(2*sigma2*sigma2)
            noise[i,j]+=np.exp(power)
   
   
    cv.imshow('Illumination effect',noise)
    
    noise=cv.normalize(noise, None, 0, 255, cv.NORM_MINMAX)
    noise = noise.astype(np.uint8)
    
    noisy_image=cv.add(image, noise)
    
    noisy_image=cv.normalize(noisy_image, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image






img = cv.imread('./lena.png', 0)

cv.imshow('Input Image before shedding',img)

img= add_gaussian_noise(img, 30, 60)

img_h, img_w = img.shape

cv.imshow('Input Image after shedding', img)



filter_homo = np.zeros((img_h, img_w), dtype=np.float32)

yh = 1.1
yl =0.3
c = 0.1/2
d0 = 50




center_i, center_j = img_h // 2, img_w // 2

for i in range(img_h):
    for j in range(img_w):
        duv = (i-center_i)**2 + (j-center_j)**2
        filter_homo[i, j] = (yh-yl) * (1-np.exp(-c * (duv / d0**2))) + yl
        
cv.imshow('Homomorphic filter', filter_homo)

img_log = np.log1p(img)
F = np.fft.fft2(img_log)
F_shift = np.fft.fftshift(F)

magnitude_spectrum=np.log(np.abs(F))
magnitude_spectrum=cv.normalize(magnitude_spectrum,None,0,255,cv.NORM_MINMAX, dtype=cv.CV_8U)
cv.imshow('Magnitude spectrum',magnitude_spectrum)

magnitude_spectrum_after_shift=np.log(np.abs(F_shift))
magnitude_spectrum_after_shift=cv.normalize(magnitude_spectrum_after_shift,None,0,255,cv.NORM_MINMAX, dtype=cv.CV_8U)
cv.imshow('Magnitude spectrum after shift',magnitude_spectrum_after_shift)








G_shift = F_shift * filter_homo

G = np.fft.ifftshift(G_shift)
output = np.fft.ifft2(G).real


output = np.exp(output) - 1


output = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

output=np.round(output).astype(np.uint8)


cv.imshow('Output',output)





cv.waitKey(0)
cv.destroyAllWindows()