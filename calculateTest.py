#!/usr/bin/env python
#-*-coding:utf-8-*-
#@File:calculateTest.py
import cv2
import numpy as np
import math

def calMSE(a,b):
    mse=0
    for i in range(1500):
        for j in range(1000):
            for k in range(3):
                mse+=(a[j][i][k]-b[j][i][k])**2
    mse=mse/1500.0/1000
    return mse

def calPSNR(mse,n):
    return 10*math.log10((math.pow(2,n)-1)**2/mse)

a=cv2.imread('result3.tif',flags=cv2.IMREAD_UNCHANGED)
b=cv2.imread('HDRImg.tif',flags=cv2.IMREAD_UNCHANGED)
print(a)
print(b)

mse=calMSE(a,b)
print('mse',mse)
print('psnr',calPSNR(mse,n=16))
