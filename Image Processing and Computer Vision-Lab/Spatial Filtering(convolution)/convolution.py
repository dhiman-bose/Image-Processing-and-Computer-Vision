import sys
sys.path.append("C:\\Users\\SSD\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")



import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def convolution(img,kernel):
    
    pad=kernel.shape[0]//2
    fp=np.pad(img, ((pad, pad), (pad,pad)),mode='constant', constant_values=0).astype(np.float32) #np.float64
    
    fo=np.zeros_like(fp)
    h=fp.shape[0]
    w=fp.shape[1]
    for x in range(pad,h-pad):
        for y in range(pad,w-pad):
            sum=0
            s=t=pad
            
            for shift_x in range(-pad,pad+1):
                for shift_y in range(-pad,pad+1):
                    sum+=(kernel[s-shift_x][t-shift_y]*fp[x+shift_x][y+shift_y])
            
            fo[x][y]=sum
            
    cv.normalize(fo,fo, 0, 255, cv.NORM_MINMAX)
    fo=np.round(fo).astype(np.uint8)
    
    return fo

def conv(img,kernel):
    
    pad=kernel.shape[0]//2
    fp=np.pad(img, ((pad, pad), (pad,pad)),mode='constant', constant_values=0).astype(np.float32) #np.float64
    
    fo=np.zeros_like(fp)
    h=fp.shape[0]
    w=fp.shape[1]
    for x in range(pad,h-pad):
        for y in range(pad,w-pad):
            sum=0
            s=t=pad
            
            for shift_x in range(-pad,pad+1):
                for shift_y in range(-pad,pad+1):
                    sum+=(kernel[s-shift_x][t-shift_y]*fp[x+shift_x][y+shift_y])
            
            fo[x][y]=sum
            
    
    
    return fo

def mean(kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel= kernel / kernel.sum()
    fo=convolution(img,kernel)
   
   
    cv.imshow('Mean Output',fo)
    
    
def GaussianFilter(img,size,sigma):
    mid = size // 2
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
    kernel=np.zeros((size,size))
    
        
                    
    for i in range(size):
        
        for j in range(size):
            
            x=i-mid;
            xx=x*x;
            y=j-mid;
            yy=y*y
            
            const=1/(2*sigma*sigma)
            kernel[i][j]=np.exp(-(xx+yy)*const)
    norm=np.sum(kernel)
    kernel=kernel/norm
    
    fo=convolution(img,kernel)
    
    cv.imshow('Gaussian Output',fo)
    








def laplacian_filter(img):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    fo=convolution(img,kernel)
    
   
    cv.imshow('Laplacian Output',fo)


def lab_filter(img):
    kernel = np.array([[-12,-11,-10,-9,-8],[-7,-6,-5,-4,-3],[-2,-1,0,1,2],[3,4,5,6,7],[8,9,10,11,12]])
    fo=convolution(img,kernel)
    
   
    cv.imshow('Lab Output',fo)    






def Median(img,kernel_size):
    
    pad=kernel_size//2
    fp=np.pad(img, ((pad, pad), (pad,pad)),mode='constant', constant_values=0).astype(np.uint8) #np.float64
    fo=np.zeros_like(fp)
    h=fp.shape[0]
    w=fp.shape[1]
    for x in range(pad,h-pad):
        for y in range(pad,w-pad):
            sum=0
            s=t=pad
            neighbours=np.array([])
            
            for shift_x in range(-pad,pad+1):
                for shift_y in range(-pad,pad+1):
                    neighbours=np.append(neighbours,fp[x+shift_x][y+shift_y])
            neighbours=np.sort(neighbours)
            median=np.median(neighbours)
            
            fo[x][y]=median
    
    cv.imshow('Median Output',fo)



def Sobel(img):
    h = img.shape[0]
    w = img.shape[1]
    kernel_x = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    kernel_y = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    pad=kernel_x.shape[0]//2
    res=np.zeros((h+pad+pad,w+pad+pad))
    
    resx = conv(img,kernel_x)
    resy = conv(img,kernel_y)
    
      
    res=np.sqrt(resx**2+resy**2)
  
    cv.normalize(resx,resx, 0, 255, cv.NORM_MINMAX)
    
    resx=np.round(resx).astype(np.uint8)
    
    
    cv.normalize(resy,resy, 0, 255, cv.NORM_MINMAX)
    
    resy=np.round(resy).astype(np.uint8)
  
    
    
    cv.normalize(res,res, 0, 255, cv.NORM_MINMAX)
    
    res=np.round(res).astype(np.uint8)
    
    
    cv.imshow('Sobelx',resx)
    cv.imshow('Sobely',resy)
    cv.imshow('Sobel Output',res)
    
    

    
    
   




#img = cv.cvtColor(cv.imread('./lena.png'),cv.COLOR_BGR2GRAY)
img = cv.imread('./lena.png',0)
print(img.shape)
cv.imshow('input',img)

mean(3)
Median(img,3)

GaussianFilter(img,5,1)
laplacian_filter(img)

Sobel(img)
lab_filter(img)


cv.waitKey(0)
cv.destroyAllWindows()