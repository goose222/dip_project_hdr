#!/usr/bin/env python
#-*-coding:utf-8-*-
#@File:aligment.py
import cv2
import numpy as np
import math

def optical_flow(frame1,frame2):
    frame2 = np.float32(np.clip(frame2, 0,frame1.max()))
    #print(frame1)
    #print(frame2)
    prvs = cv2.cvtColor(np.float32(frame1),cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame2', prvs/prvs.max())
    #cv2.waitKey(30000)
    next = cv2.cvtColor(np.float32(frame2), cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame2', next/next.max())
    #cv2.waitKey(30000)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 8, 6, 1, 5, 1.1,0)
    print(flow)
    return flow

def BiBubic(x):
    x=abs(x)
    if x <= 1:
        return 1 - 2 * (x ** 2) + (x ** 3)
    elif x < 2:
        return (4 - 8 * x + 5 * (x ** 2) - (x ** 3))
    else:
        return 0

def BiCubic_interpolation(img, flow):
    scrH, scrW, _ = img.shape
    retimg = np.zeros((scrH, scrW, 3))
    for i in range(scrH):
        for j in range(scrW):
            retimg[i, j] = img[i][j]
            scrx =  i + flow[i][j][0]
            scry =  j + flow[i][j][1]
            x = int(round(scrx,0))
            y = int(round(scry,0))
            '''u = scrx - x
            v = scry - y
            tmp = 0
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    if x + ii < 0 or y + jj < 0 or x + ii >= scrH or y + jj >= scrW:
                        continue
                    tmp = tmp+ img[x + ii, y + jj] * BiBubic(ii - u) * BiBubic(jj - v)
            retimg[i, j] = tmp
            '''
            if x<0:
                x=0
            if x>=scrH:
                x=scrH-1
            if y<0:
                y=0
            if y>=scrW:
                y=scrW-1
            retimg[i, j] = img[x, y]
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    if i+ii>=0 and i+ii<scrH and j+jj>=0 and j+jj<scrW and x+ii<scrH and y+jj<scrW:
                        retimg[i+ii, j+jj] = img[x+ii, y+jj]
    return retimg


