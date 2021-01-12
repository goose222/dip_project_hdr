#!/usr/bin/env python
#-*-coding:utf-8-*-
#@File:test.py
import cv2
import numpy as np
import math
import os
import matplotlib.image as mpimg
import torchvision.transforms as transforms

import torch
import numpy as np
from torch.autograd import Variable
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torch import nn
from torch import optim

def getImage(filepath,expo):
    image=[]
    for i in range(4):
        if i<9:
            dir=filepath+"00"+str(i+1)+'/'
        else:
            dir=filepath+"0"+str(i+1)+'/'
        imagelist = os.listdir(dir)
        image3=[]
        for imgname in imagelist:
            if (imgname.endswith(".tif")):
                image1 = cv2.imread(dir + imgname,flags = cv2.IMREAD_UNCHANGED)
                image3.append(image1)
        for j in range(3):
            image3[j]=np.float32(image3[j])/65535.0
            image3[j]=np.power(image3[j],2.2)/(expo[i][j])
        image.append(image3)
    print(len(image))
    return image
    
def getExpo(filepath):
    expo=[]
    for i in range(4):
        if i<9:
            dir=filepath+"00"+str(i+1)+'/'
        else:
            dir=filepath+"0"+str(i+1)+'/'
        with open(dir+'exposure.txt', "r") as f:  
            data= f.readlines()
            expo1=[]
            for j in range(len(data)):
                if float(data[j][0:-1])==0:
                    expo1.append(1)
                else:
                    expo1.append(math.pow(2,int(float(data[j][0:-1]))))
            expo.append(expo1)
    return expo

def getResult(filepath):
    result=[]
    for i in range(4):
        if i<9:
            dir=filepath+"00"+str(i+1)+'/'
        else:
            dir=filepath+"0"+str(i+1)+'/'
        img = cv2.imread(dir+"HDRImg.hdr", flags = cv2.IMREAD_ANYDEPTH)
        result.append(img)
    return result

def convert(output):
  output=output.swapaxes(3,1)
  output=output.swapaxes(3,2)
  return output


def convert2(inputt):
  input1=[]
  for i in range(3):
    for j in range(3):
      input1.append(inputt[i,:,:,:,j])
  input1=np.array(input1)
  input1=input1.swapaxes(1,0)
  return input1
  
def cutImage(image):
    result=[]
    for i in image:
        #print(i.shape)
        schy=0
        while schy<=1200:
            schx = 0
            while schx <= 700:
                if schy == 1200 and schx == 700:
                    a = i[schx:,schy:]
                elif schy == 1200:
                    a = i[schx:schx + 300,schy:]
                elif schx == 700:
                    a = i[schx:,schy:schy + 300]
                else:
                    a= i[schx:schx + 300,schy:schy+300]
                result.append(a)
                schx+=100
            schy+=300
    print(len(result))
    return result
#####################################################
#####################################################
filepath="/home/linduaner/cnn/Training/"
expo = getExpo(filepath)
inputt = np.array(getImage(filepath,expo))
del expo
output=getResult("/home/linduaner/cnn/Training/")
output=np.array(cutImage(output))

inputt1=[]
for x in range(3):
  inputt1.append(np.array(cutImage(inputt[:,x])))
  
inputt1=np.float32(inputt1)
inputt=inputt1
del inputt1

output=np.float32(output)

output=convert(output)
inputt=convert2(inputt)
print(inputt.shape)
print(output.shape)
inputt=torch.from_numpy(inputt)
output=torch.from_numpy(output)

train_dataset=TensorDataset(inputt,output)
train_loader = DataLoader(dataset = train_dataset,
                         batch_size=50,
                         shuffle=False)
                         


class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=9,  # (9,300,300)
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=1), 
            torch.nn.BatchNorm2d(16),  
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)  # (9,150,150)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)  # 2x2采样，output shape (32,75,75)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1,
                                     output_padding=0, bias=True),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )  # (16,200,300)
        self.conv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=9, kernel_size=3, stride=1, padding=1,
                                     output_padding=0, bias=True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU(),
        )  # (9,1000,1500)
        self.conv5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=18, out_channels=3, kernel_size=3, stride=1, padding=1,
                                     output_padding=0, bias=True),
            torch.nn.BatchNorm2d(3),
            #torch.nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        
        x3 = self.conv3(x2)
        x3=torch.nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True),
        xx = torch.cat((x3[0], x1), 1)
        
        x4 = self.conv4(xx)
        x4=torch.nn.functional.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True),
        xx = torch.cat((x4[0], x), 1)
        
        x = self.conv5(xx)

        return x


model = CNNnet()
model = torch.load('model2.pkl')
model.cpu().eval()
              

num=0            
for i, data in enumerate(train_loader):
  inputs, outputs = data  
  inputs, outputs = Variable(inputs), Variable (outputs)
  pred_y = model(inputs)
  
  y=pred_y.detach().numpy()
  print(y.shape)
  for x in y:
    x=np.array(x)
    print(x.shape)
    x=x.swapaxes(2,0)
    x=x.swapaxes(1,0)
    cv2.imwrite("-now-"+str(num) + '.hdr',x)
    num+=1