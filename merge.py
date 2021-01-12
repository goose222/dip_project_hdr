#!/usr/bin/env python
#-*-coding:utf-8-*-
#@File:merge.py
import cv2
import numpy as np
import math
import os

filename='C:/Users/63093/Desktop/3/'
def getImage(filepath):
    image=[]
    for i in range(40):
        dir=filepath+'-now-'+str(i+80)+'.hdr'
        print(dir)
        img = cv2.imread(dir, flags = cv2.IMREAD_UNCHANGED)
        image.append(np.array(img))
        print(img.shape)
    print(len(image))
    return image

images=np.array(getImage(filename))
#print(images)

cols=[]
i=0
while i<8:
    tmp=images[i]
    for j in range(4):
        x=i+j*8+8
        #print(x)
        tmp=np.append(tmp,images[x],axis=1)
    cols.append(tmp)
    #cv2.imshow('test',np.array(cols[i]))
    #cv2.waitKey(1000)
    print(i)
    i = i + 1


cols=np.array(cols)
num=np.zeros((1000,1500,3))
print(num.shape)
num[0:100,:]=cols[0,0:100]
num[100:200,:]=(cols[0,100:200]+cols[1,0:100])/2
num[200:300]=(cols[0,200:]+cols[1,100:200]+cols[2,0:100])/3
num[300:400]=(cols[1,200:]+cols[2,100:200]+cols[3,0:100])/3
num[400:500]=(cols[2,200:]+cols[3,100:200]+cols[4,0:100])/3
num[500:600]=(cols[3,200:]+cols[4,100:200]+cols[5,0:100])/3
num[600:700]=(cols[4,200:]+cols[5,100:200]+cols[6,0:100])/3
num[700:800]=(cols[5,200:]+cols[6,100:200]+cols[7,0:100])/3
num[800:900]=(cols[6,200:]+cols[7,100:200])/2
num[900:]=cols[7,200:]

cv2.imshow('test',np.array((num)**(1/2.2)))
cv2.imwrite('result.hdr',np.array(num))
cv2.waitKey(10000)
