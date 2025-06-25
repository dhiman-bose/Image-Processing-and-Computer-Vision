


import sys
sys.path.append("C:\\Users\\SSD\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")


import cv2 as cv
import numpy as np







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
    
    result = (1-a)*(1-b)*I[j,k]+a*(1-b)*I[j+1,k]+(1-a)*b*I[j,k+1]+a*b*I[j+1,k+1]
    
    return result



def angular(img,a,t):
    
   
    h=img.shape[0]
    w=img.shape[1]
    xc,yc=h//2,w//2
    out = np.zeros((h,w),dtype=np.uint8)
    
    
    
           
    for i in range(h):
        for j in range(w):
            dx=i-xc
            dy=j-yc
            
            r=np.sqrt((dx**2)+(dy**2))
            
            
            dx=np.deg2rad(dx)
            dy=np.deg2rad(dy)
            beta=np.arctan2(dy,dx)+a*(np.sin((2*np.pi*r)/t))  
            #beta=dx+a*(np.sin((2*np.pi*r)/t))     
            x = xc + r*np.cos(beta)
            
            
            #beta=dy+a*(np.sin(2*np.pi*r/t))
            y = yc + r*np.sin(beta)
            
                
            
            out[i][j] = interpolation(img,x,y)
    
    return out
  

  

img = cv.imread('inp.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, img = cv.threshold(img,100,255,cv.THRESH_BINARY)

cv.imshow('Input Image',img)

a=0.1
t=50

img_t=angular(img,a,t)


cv.imshow('Output Image for a=0.1',img_t)


a=0.05
t=50

img_t=angular(img,a,t)


cv.imshow('Output Image for a=0.05 ',img_t)





cv.waitKey(0)
cv.destroyAllWindows()