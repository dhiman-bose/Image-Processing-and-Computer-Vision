
import sys
sys.path.append("C:\\Users\\SSD\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def equalization(img,name):
    img_h, img_w = img.shape
    L = 256
    M, N = img_h, img_w
    
    n_k = cv.calcHist([img],[0],None,[256],[0,255])
    title=" Histogram before equalization "+name+" channel"
    plt.title(title)
    #plt.plot(n_k)
    plt.hist(img.ravel(),0,(0,255))
    plt.figure(figsize=(10,4))
    
    
    
    p_r = n_k / (M*N)

    cdf = np.zeros(L)

    for k in range(L):
        cdf[k] = np.sum(p_r[:k+1])

    s_k = (L-1) * cdf

    s_k = np.round(s_k)
    '''
    title=" Histogram after equalization "+name+" channel"
    plt.title(title)
    plt.plot(s_k)
    plt.figure(figsize=(10,4))
    '''
    plt.show()
    output = np.zeros((img_h, img_w), dtype=np.uint8)

    for i in range(img_h):
        for j in range(img_w):
            output[i, j] = s_k[img[i, j]]
    
    
    title=" Histogram after equalization "+name+" channel"
    plt.title(title)
    #plt.plot(s_k)
    plt.hist(output.ravel)
    plt.figure(figsize=(10,4))
    
    return output





img = cv.imread('color_img.jpg')
#img=cv.cvtColor(img, cv.COLOR_BGR2RGB)

cv.imshow('Input Image',img)
b,g,r = cv.split(img)

r=equalization(r,'r')
g=equalization(g,'g')
b=equalization(b,'b')

new=cv.merge((b,g,r))
cv.imshow('Output Image by equalizing each channels',new)

img_hsi = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h,s,i=cv.split(img_hsi)
i=equalization(i,'i')

hsi=cv.merge((h,s,i))
hsi = cv.cvtColor(hsi, cv.COLOR_HSV2BGR)

cv.imshow('Output Image by equalizing intensity',hsi)




cv.waitKey(0)
cv.destroyAllWindows()



