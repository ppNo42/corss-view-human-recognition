import data_process.get_dataset as mydata
import torchvision
from torch.utils.data import DataLoader
import torch
from torch import nn
from d2l import torch as d2l

p = mydata.MyDataset("train",18,(179,180),'nm')
print(len(p),p[35][0].shape,p.positive_sample[35],p[35][1])

test_loader = DataLoader(p, batch_size=16, shuffle=False, num_workers=0, drop_last=True)

for data in test_loader:
    print(data[0].shape,data[1])

net = nn.Sequential(
    nn.Conv2d(2, 16, kernel_size=7, stride=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(16, 64, kernel_size=7,stride=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 256, kernel_size=7,stride=1),
    nn.Dropout(),
    nn.Flatten(),
    nn.Linear(256*21*11, 2))

nn.BCELoss()

d2l.astype()
d2l.accuracy()


import fastai
fastai.callbacks



