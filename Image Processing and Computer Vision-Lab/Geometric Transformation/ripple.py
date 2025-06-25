
import sys
sys.path.append("C:\\Users\\SSD\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt







def interpolation(I, x, y):
    #bilinear interpolation
    h=I.shape[0]
    w=I.shape[1]
    
    j=int(np.floor(x))
    k=int(np.floor(y))
    j=min(j,h-2)
    k=min(k,w-2)
    
    b=x-j
    a=y-k
    
    result = (1-a)*(1-b)*I[j,k]+a*(1-b)*I[j+1,k]+(1-a)*b*I[j,k+1]+a*b*I[j,k]
    return result



def ripple(img):
    output = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    for k in range(img.shape[2]):
        h=img[:,:,k].shape[0]
        w=img[:,:,k].shape[1]
        out = np.zeros((h,w),dtype=np.float32)
       
      
        
        for i in range(h):
            for j in range(w):
                
                x = abs(i + ax*np.sin(2*np.pi*j/tx))
                y = abs(j + ay*np.sin(2*np.pi*i/ty))
                
                
                out[i][j] = interpolation(img[:,:,k],x,y)
        cv.normalize(out,out,0,255,cv.NORM_MINMAX)
        out=np.round(out).astype(np.uint8)
        
        output[:,:,k]=out
    return output
  

  

img = cv.imread('ripple.jpg')
img_g = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

ax=10
ay=10
tx=20
ty=20


img_t=ripple(img)
cv.imshow('Input Image',img)

cv.imshow('Output Image for ax=ay=10',img_t)


ax=10
ay=15
tx=50
ty=70
img_t=ripple(img)
cv.imshow('Input Image',img)

cv.imshow('Output Image for ax=10,ay=15',img_t)




cv.waitKey(0)
cv.destroyAllWindows()