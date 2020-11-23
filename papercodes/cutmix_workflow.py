import numpy as np
#from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from time import time
#from sklearn.metrics import confusion_matrix
import io
from PIL import Image as pil_image
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


model = torch.nn.Sequential(OrderedDict([
                ('conv_base',torch.hub.load('pytorch/vision:v0.6.0', 'densenet201', pretrained=True).features[:-1]),
                ('globalavgpool',torch.nn.AvgPool2d(kernel_size = 7)),
                ('flatten',torch.nn.Flatten()),
                ('lastlinear',torch.nn.Linear(1920,2))
            ])
        )


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train(model):
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        ])

    train_data = datasets.ImageFolder(
        'randimages/train',
        transform = train_transforms
    )

    #print(train_data.class_to_idx)

    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size = 34,
        shuffle = True
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('randimages/val', transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            ])),
        batch_size = 231,
    )

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    epochs = 2

    
    for i in range(epochs):
        model.train()
        for images, labels in trainloader:
            images/=255.
            r = np.random.rand(1)
            if r<0.5:
                rand_index = torch.randperm(images.shape[0])
                target_a = labels
                target_b = labels[rand_index]
                lam = np.random.beta(1, 1)
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:,:,bbx1:bbx2,bby1:bby2] = images[rand_index,:,bbx1:bbx2,bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                output = model(images)
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
                print('train loss:{}'.format(loss), end = ' ')
            else:
                output = model(images)
                loss = criterion(output, target)
                print('train loss:{}'.format(loss), end = ' ')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        #Validation
        model.eval()
        for images, labels in val_loader:
            images/=255.
            op = model(images)
            val_loss = criterion(op,labels)
            print('val loss:{}'.format(val_loss))
    


train(model)

"""
print(model.classifier.in_features) 
print(model.classifier(a).shape)
"""
"""
img = 'a'
with open('10.tif', 'rb') as f:
        img = pil_image.open(io.BytesIO(f.read()))
img = img.resize((256,256),pil_image.NEAREST)

img.show()

from tensorflow.keras.preprocessing.image import img_to_array
imgnp = img_to_array(img)
print(imgnp.shape)
print(img.mode)
"""



