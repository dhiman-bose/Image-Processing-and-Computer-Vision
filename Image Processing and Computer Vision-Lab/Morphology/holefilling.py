import sys
sys.path.append("C:\\Users\\SSD\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")
import numpy as np
import cv2 as cv



def _dilation(img, kernel):
    padx = kernel.shape[0] // 2
    pady = kernel.shape[1] // 2
    
    img = np.pad(img, ((padx, padx), (pady, pady)), mode='constant', constant_values=0).astype(np.uint8)
    
    h, w = img.shape
    

    op = np.zeros_like(img)

    for x in range(padx, h - padx):
        for y in range(pady, w - pady):
            flag = 0
            
            for i in range(-padx, padx + 1):
                for j in range(-pady, pady + 1):
                    if kernel[i + padx, j + pady] == 1 and img[x + i, y + j] == 1:
                        flag = 1

            if flag:
                op[x, y] = 1
            else:
                op[x, y] = 0
           
    return op[padx:h-padx,pady:w-pady]






#kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
kernel = np.zeros((3,3), np.uint8)
kernel[0,0]=kernel[0,2]=kernel[2,0]=kernel[2,2]=0
kernel[:,1]=kernel[1,:]=1

print(kernel)






img = np.zeros((10,12), np.uint8)
img[1,3:6]=img[1:8,5]=img[9,1:7]=img[7:10,6] =  img[7,6]= img[4:8, 2]=img[3,1]=img[2,2]=img[8,1]=  1




X = np.zeros_like(img)
X[7,4]=1


count=0





'''

ratex = 512/10
ratey = 512/12
img = cv.resize(img, None, fx = ratey, fy = ratex, interpolation = cv.INTER_NEAREST)
X= cv.resize(X, None, fx = ratey, fy = ratex, interpolation = cv.INTER_NEAREST)
'''

Ac = np.abs(1 - img)


#print(I,ac,k,x)
while(True):
    count=count+1
    prevX = X
    
    #a=cv.dilate(X,kernel,iterations=1)
    a=_dilation(X,kernel)
   
    X = np.bitwise_and(a, Ac)
    if np.array_equal(X, prevX):
        break

  

X=np.bitwise_or(X,img)


ratex = 512/10
ratey = 512/12
img = cv.resize(img, None, fx = ratey, fy = ratex, interpolation = cv.INTER_NEAREST)
X= cv.resize(X, None, fx = ratey, fy = ratex, interpolation = cv.INTER_NEAREST)



ratex = 150/3
ratey = 150/3
kernel = cv.resize(kernel, None, fx = ratey, fy = ratex, interpolation = cv.INTER_NEAREST)

cv.imshow('input',img*255)
cv.imshow('kernel',kernel*255)
cv.imshow('X',X*255)

cv.waitKey(0)
cv.destroyAllWindows()
