import numpy as np
#from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from time import time
#from sklearn.metrics import confusion_matrix
import io
from PIL import Image as pil_image
import torch
from torchvision import datasets, transforms


"""
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


def train():
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
        ])

    train_data = datasets.ImageFolder(
        'randimages',
        transform = train_transforms
     )

    #print(train_data.class_to_idx)

    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size = 34,
        shuffle = True
    )

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    epochs = 2
    for i in range(epochs):
        for images, labels in trainloader:
            r = np.random.rand(1)
            if r<0.5:
                randindex = torch.randperm(images.shape[0])
                target_a = target
                target_b = target[rand_index]
                lam = np.random.beta(1, 1)
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:,:,bbx1:bbx2,bby1:bby2] = images[randindex,:,bbx1:bbx2,bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                output = model(images)
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            else:
                output = model(input)
                loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
"""        
a = torch.randn(1,3,224,224)
model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)


for name,child in model.named_children():
    print(child)

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


#print(img.mode)
"""



