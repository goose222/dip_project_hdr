#!/usr/bin/env python
#-*-coding:utf-8-*-
#@File:test.py
import cv2
import numpy as np
import math
import os
import torchvision.transforms as transforms

import torch
from torch.autograd import Variable
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torch import nn
from torch import optim


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ids = 0

def getImage(filepath,expo):
    image=[]
    for i in range(74):
        if i<9:
            dir=filepath+"00"+str(i+1)+'/'
        else:
            dir=filepath+"0"+str(i+1)+'/'
        imagelist = os.listdir(dir)
        image3=[]
        for imgname in imagelist:
            if (imgname.endswith(".tif")):
                image1 = cv2.imread(dir + imgname,flags = cv2.IMREAD_UNCHANGED)
                image3.append(np.array(image1))
        for j in range(3):
            image3[j]=np.float32(image3[j])/65535.0
            image3[j]=np.power(image3[j],2.2)/(expo[i][j])
        image.append(image3)
    print(len(image))
    return np.array(image)
    
def getExpo(filepath):
    expo=[]
    for i in range(74):
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
    for i in range(74):
        if i<9:
            dir=filepath+"00"+str(i+1)+'/'
        else:
            dir=filepath+"0"+str(i+1)+'/'
        img = cv2.imread(dir+"HDRImg.hdr", flags = cv2.IMREAD_UNCHANGED)
        result.append(np.array(img))
    return np.array(result)

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
        while schy<=1250:
            schx = 0
            while schx <= 750:
                if schy == 1250 and schx == 750:
                    a = i[schx:,schy:]
                elif schy == 1250:
                    a = i[schx:schx + 250,schy:]
                elif schx == 750:
                    a = i[schx:,schy:schy + 250]
                else:
                    a= i[schx:schx + 250,schy:schy+250]
                result.append(a)
                #print(a.shape)
                for h in range(3):
                    M = cv2.getRotationMatrix2D((125,125),90,1)
                    a = cv2.warpAffine(a, M, (250, 250))
                    result.append(np.array(a))
                schx+=100
            schy+=100
    print(len(result))
    return result
#####################################################
filepath="/home/linduaner/cnn/Training/"
expo = getExpo(filepath)
inputt = getImage(filepath,expo)
del expo
print(inputt.shape)
output=getResult("/home/linduaner/cnn/Training/")
output=np.array(cutImage(output))

inputt1=[]
for x in range(3):
  inputt1.append(np.array(cutImage(inputt[:,x])))
  
del inputt
inputt1=np.float32(inputt1)
print(inputt1.shape)
print(output.shape)
inputt=inputt1
del inputt1

output=np.float32(output)

output=convert(output)
inputt=convert2(inputt)

inputt=torch.from_numpy(inputt)
output=torch.from_numpy(output)

train_dataset=TensorDataset(inputt,output)
train_loader = DataLoader(dataset = train_dataset,
                         batch_size=37,
                         shuffle=True)
del inputt
del output
#####################################################   

#######################################################
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,  
                            out_channels=8,
                            kernel_size=3,
                            stride=1,
                            padding=2), 
            torch.nn.BatchNorm2d(16),  ##200*200 *16 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)  # (24,100,100)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, 3, 1, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)  #  (48,50,50)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)  #(32*3,25,25)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1,
                                     output_padding=0, bias=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )  # (32,50,75)
        self.conv5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=80, out_channels=16, kernel_size=3, stride=1, padding=1,
                                     output_padding=0, bias=True),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )  # (16,100,150)
        self.conv6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=40, out_channels=9, kernel_size=3, stride=1, padding=1,
                                     output_padding=0, bias=True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU(),
        )  # (9,200,300)
        self.conv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=18, out_channels=3, kernel_size=3, stride=1, padding=1,
                                     output_padding=0, bias=True),
            torch.nn.BatchNorm2d(3),
            #torch.nn.ReLU(),
        )

        

    def forward(self, x):
        x11 = self.conv1(x[0:3])
        x12 = self.conv1(x[3:6]) 
        x13 = self.conv1(x[6:])
        
        x21 = self.conv2(x11)
        x22 = self.conv2(x12)
        x23 = self.conv2(x13)
        
        x31 = self.conv3(x21)
        x32 = self.conv3(x22)
        x33 = self.conv3(x23)
        
        x1=torch.cat((x11, x12,x13), 1)
        x2=torch.cat((x21, x22,x23), 1)
        x3=torch.cat((x31, x32,x33), 1)
        
        x4 = self.conv4(x3)
        x4=torch.nn.functional.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True),
        xx = torch.cat((x4[0], x2), 1)
        
        x5 = self.conv5(xx)
        x5=torch.nn.functional.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True),
        xx = torch.cat((x5[0], x1), 1)
        
        x6 = self.conv6(xx)
        x6=torch.nn.functional.interpolate(x6, scale_factor=2, mode='bilinear', align_corners=True),
        xx = torch.cat((x6[0],x), 1)
        
        x7 = self.conv7(xx)

        return x


model = CNNnet()

class My_loss(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        
    def forward(self, x, y):
        x1 = torch.log2(1 + x * 5000) / math.log(1 + 5000,2)  # T tonemapping
        y1 = torch.log2(1 + y * 5000) / math.log(1 + 5000,2)  # T tonemapping
        return torch.sum(torch.pow((x - y), 2))


#model = torch.load('\model.pkl')
opt = torch.optim.Adam(model.parameters(),lr=0.001)
loss_func = My_loss() 

for epoch in range(1):
    for i, data in enumerate(train_loader):
        model.train()
        inputs, outputs = data
        inputs=Variable(inputs,requires_grad=True)
        outputs = Variable (outputs)
        print "epoch:" ,epoch," i:" , i, "inputs", inputs.data.size(), "labels", outputs.data.size()
        pred_y = model(inputs)
        loss = loss_func(pred_y, outputs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.save(model, '\model.pkl')
        print('loss:',loss)


  
  
  
  