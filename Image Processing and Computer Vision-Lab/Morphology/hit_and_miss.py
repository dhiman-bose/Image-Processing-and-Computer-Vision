import sys
sys.path.append("C:\\Users\\SSD\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def hit_and_miss(img,k):
    img_c=255-img
    w = np.ones((3,3),np.uint8)
    kc=w-k
    k=k*255
    kc=kc*255
    
  
    
    
    rate = 50
    
    k = cv.resize(k, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
    kc = cv.resize(kc, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
    
    cv.imshow('k',k)
    cv.imshow('kc',kc)
    
    
    x=cv.erode(img,k,iterations = 1)
    z=cv.erode(img_c,kc,iterations = 1)
    op = np.bitwise_and(x,z)
    
    return op






img = cv.imread("input.jpg", cv.IMREAD_GRAYSCALE)
cv.imshow('input',img)


t, img = cv.threshold(img,100,255,cv.THRESH_BINARY)
img = img.astype(np.uint8)



cv.imshow("Thresh",img)
img_c=255-img
cv.imshow('Image_c',img_c)



k1=np.array([[0,0,0],[1,1,0],[1,0,0]],dtype=np.uint8)
op1=hit_and_miss(img, k1)
cv.imshow('Output for k1',op1)




k2=np.array([[0,1,1],[0,0,1],[0,0,1]],dtype=np.uint8)
op2=hit_and_miss(img, k2)
cv.imshow('Output for k2',op2)


k3=np.array([[1,1,1],[0,1,0],[0,1,0]],dtype=np.uint8)
op3=hit_and_miss(img, k3)
cv.imshow('Output for k3',op3)

cv.waitKey(0)
cv.destroyAllWindows()