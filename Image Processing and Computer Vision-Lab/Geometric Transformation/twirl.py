



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
    
    result = (1-a)*(1-b)*I[j,k]+a*(1-b)*I[j+1,k]+(1-a)*b*I[j,k+1]+a*b*I[j,k]
    return result



def twirl(img,alpha):
    
   
    h=img.shape[0]
    w=img.shape[1]
    xc,yc=h//2,w//2
    out = np.zeros((h,w),dtype=np.uint8)
    
    
    
    rmax=175
           
    print(rmax)
    
    for i in range(h):
        for j in range(w):
            dx=(i-xc)
            dy=(j-yc)
            
            r=np.sqrt((dx**2)+(dy**2))
            if(r<=rmax):
                
                beta=np.arctan2(dy,dx)+alpha*((rmax-r)/rmax)
               
                x = xc + r*np.cos(beta)
                y = yc + r*np.sin(beta)
            else:
                x,y=i,j
                
            
            out[i][j] = interpolation(img,x,y)
            
    
    return out
  

  

img = cv.imread('inp.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, img = cv.threshold(img,100,255,cv.THRESH_BINARY)

cv.imshow('Input Image',img)

alpha=90

alpha=np.deg2rad(alpha)

#alpha=alpha*np.pi/180

img_t=twirl(img,alpha)


cv.imshow('Output Image for alpha=90 degree',img_t)

alpha=-90
alpha=np.deg2rad(alpha)

#alpha=alpha*np.pi/180


img_t=twirl(img,alpha)


cv.imshow('Output Image for alpha=-90 degree',img_t)





cv.waitKey(0)
cv.destroyAllWindows()
