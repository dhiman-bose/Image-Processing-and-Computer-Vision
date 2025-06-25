

import sys

sys.path.append("C:\\Users\\SSD\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


img = cv.imread('two_noise.jpeg',cv.IMREAD_GRAYSCALE)
img_h, img_w = img.shape
center_i, center_j = img_h//2, img_w//2

cv.imshow('Image',img)
matplotlib.use('TkAgg')
point_list=[]
u=[]
v=[]

x = None
y = None

def onclick(event):
    global x, y
    ax = event.inaxes
    if ax is not None:
        x, y = ax.transData.inverted().transform([event.x, event.y])
        x = int(round(x))
        y = int(round(y))
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, x, y))
        print(center_i, center_j)
        
        
       
        di=center_i-x
        dj=center_j-y
        u.append(center_i+di)
        v.append(center_j+dj)
        
        u.append(x)
        v.append(y)
        
       






plt.title("Please select seed pixel from the input")
im = plt.imshow(img, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)



print(u,v)



F = np.fft.fft2(img)
F_shift = np.fft.fftshift(F)

magnitude = np.log(np.abs(F))

magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
cv.imshow('Magnitude',magnitude)


magnitude_shift = np.log(np.abs(F_shift))
magnitude_shift = cv.normalize(magnitude_shift, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
cv.imshow('Magnitude after shift',magnitude_shift)


filter_butter = np.zeros((img_h, img_w), dtype=np.float32)

n = 2
d0 =5






for i in range(img_h):
    for j in range(img_w):
        prod = 1
        for k in range(len(v)):
            duv = np.sqrt((i - center_i - (u[k]-center_i))**2 + (j - center_j - (v[k]-center_j))**2)
            dmuv = np.sqrt((i - center_i + (u[k]-center_i))**2 + (j - center_j + (v[k]-center_j))**2)
            if duv != 0 and dmuv != 0:
                prod *= (1 / (1 + (((d0 / duv)**(2*n))))) * ((1 / (1 + ((d0 / dmuv)**(2*n)))))
        filter_butter[i, j] = prod
        
        
        
cv.imshow('Notch filter',filter_butter)

G_shift = F_shift * filter_butter

G = np.fft.ifftshift(G_shift)
output = np.fft.ifft2(G).real
output = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
cv.imshow('Output',output)

cv.waitKey(0)
cv.destroyAllWindows()
