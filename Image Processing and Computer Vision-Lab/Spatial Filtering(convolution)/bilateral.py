

import sys
sys.path.append("C:\\Users\\SSD\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def Filter(size):                          
    kernel=np.zeros((size,size),dtype=np.float32)
    d=size//2
    
    
    
    for i in range(size):
        for j in range(size):
            
            x=i-d
            xx=x*x  
            y=j-d
            yy=y*y
            r=np.sqrt(xx+yy)
            rd=r/d
            rd=rd*rd
            
            if rd <=1:
                
                kernel[i,j]=abs(1-rd)
            else:
                kernel[i,j]=0
    
    return kernel

def GaussianFilter(size,sigma):
    mid = size // 2
    kernel=np.zeros((size,size))
    
        
                    
    for i in range(size):
        
        for j in range(size):
            
            x=i-mid
            xx=x*x
            y=j-mid
            yy=y*y
            
            const=1/(2*sigma*sigma)
            kernel[i][j]=np.exp(-(xx+yy)*const)
            
            
    
    return kernel

def get_kernel(img, x, y, size, sigma):
    kernel = np.zeros((size, size), dtype=np.float32)
    mid = size // 2
    s=t=mid
    
    for shift_x in range(-mid, mid+1):
        for shift_y in range(-mid, mid+1):
           
            
            kernel[s+shift_x, t+shift_y] = np.exp(- (img[x, y] - img[x+shift_x, y+shift_y])**2 / (2 * sigma**2))
     
    return kernel

def convolution(img):
    size=7
    kernel=GaussianFilter(size,1)
    #kernel=Filter(size)
    pad=size//2
    
    fp=np.pad(img, ((pad, pad), (pad,pad)),mode='constant', constant_values=0).astype(np.float32) #np.float64
    
    fo=np.zeros_like(fp).astype(np.float32)
    h=fp.shape[0]
    w=fp.shape[1]
    for x in range(pad,h-pad):
        for y in range(pad,w-pad):
            kernel = kernel * get_kernel(fp, x, y, size,80)
            
            sum=0
            s=t=pad
            
            for shift_x in range(-pad,pad+1):
                for shift_y in range(-pad,pad+1):
                    sum+=(kernel[s+shift_x][t+shift_y]*fp[x-shift_x][y-shift_y])
            
            fo[x][y]=sum
    
    fo=cv.normalize(fo, fo, 0, 255, cv.NORM_MINMAX)
    fo = fo.astype(np.uint8)
    
    cv.imshow('output',fo)
    
    




img = cv.imread('./cube.png',0)
print(img.shape)
cv.imshow('input',img)


convolution(img)



cv.waitKey(0)
cv.destroyAllWindows()