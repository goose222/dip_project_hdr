#!/usr/bin/env python
#-*-coding:utf-8-*-
#@File:single.py
import cv2
import numpy as np
from sklearn import metrics
import math


a=cv2.imread('262A2629.tif',flags=cv2.IMREAD_UNCHANGED)
b=cv2.imread('262A2630.tif',flags=cv2.IMREAD_UNCHANGED)
c=cv2.imread('262A2631.tif',flags=cv2.IMREAD_UNCHANGED)



images=[a,b,c]

alignMTB = cv2.createAlignMTB()  # 对齐
alignMTB.process(images, images)

mergeMertens = cv2.createMergeMertens()
exposureFusion = mergeMertens.process(images)

cv2.imwrite('result3.hdr',exposureFusion)

