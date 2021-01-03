from PIL import Image as pil_image
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet201
import time
import numpy as np
import os
from cmatrix import getMetrics
from tqdm import tqdm
import math
from datetime import datetime
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import augmentations

torch.manual_seed(123)
np.random.seed(0)

f = 1

rootdir = 'randimages/'
#rootdir = 'dataset/6000samples/fold{}'.format(f)+'/'

trainsetsize = len(os.listdir(rootdir + 'Train/True')) + len(os.listdir(rootdir + 'Train/False'))
validationsetsize = len(os.listdir(rootdir + 'Validation/True')) + len(os.listdir(rootdir + 'Validation/False'))

trainbatchsize = 64
validationbatchsize = 64

####################################  Calculating mean and std for normalization #############
preprocess_mean_std_calculation = transforms.Compose([
	transforms.Resize(size = (256,256), interpolation = 4),
	transforms.ToTensor()
])


train_data = datasets.ImageFolder(
	rootdir + 'Train',
	transform = preprocess_mean_std_calculation
)

traindata_mean_std_loader = torch.utils.data.DataLoader(
	train_data,
	batch_size = trainsetsize
)

trainmean = 0.
trainstd = 0.

for samples, labels in traindata_mean_std_loader:
	samples = samples.reshape(3,-1)
	trainmean = samples.mean(dim = 1)
	trainstd = samples.std(dim = 1)
	break

##########################################################################################


############################ AUGMIX #####################
def normalize(image):
    image = image.transpose(2, 0, 1)
    mean, std = np.array(trainmean), np.array(trainstd)
    image = (image - mean[:, None, None])/std[:, None, None]
    return image.transpose(1, 2, 0) 


def apply_op(image, op, severity):
    image = np.clip(np.asarray(image) * 255., 0., 255.).astype(np.uint8)
    pil_img = pil_image.fromarray(image)
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img)/255.


def aug(image, preprocess,  severity = 3, width = 3, depth = -1, alpha = 1.):
    ws = np.float32(np.random.dirichlet([alpha] * width))
    ws = torch.from_numpy(ws)
    m = np.float32(np.random.beta(alpha, alpha))
    mix = torch.zeros_like(preprocess(image))
    auglist = augmentations.augmentations
    for i in range(width):
        image_aug = image.copy()
        if depth>0:
        	d = depth
        else:
        	d = np.random.randint(1,4)
        for _ in range(d):
            op = np.random.choice(auglist)
            image_aug = apply_op(image_aug, op, severity)
        mix+=ws[i] * preprocess(image_aug)

    mixed = (1-m)*preprocess(image) + m*mix
    return mixed

################################################################

class AugMixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess


    def __getitem__(self, i):
        x, y = self.dataset[i]
        im_tuple = (self.preprocess(x), aug(x, self.preprocess), aug(x, self.preprocess))
        return im_tuple, y

    def __len__(self):
        return len(self.dataset)



preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(trainmean, trainstd)])

train_transform = transforms.Compose([
	transforms.Resize(size = (256,256), interpolation = 4),
    ])

train_dataset = datasets.ImageFolder(
	rootdir + 'Train',
        train_transform
)

train_dataset = AugMixDataset(train_dataset, preprocess)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = trainbatchsize,
        shuffle = True
) 

validation_dataset = datasets.ImageFolder(
        rootdir + 'Validation',
)

validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size = validationbatchsize,
        shuffle = False
)

class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer = densenet201(pretrained = False)
		self.layer.classifier = nn.Linear(int(self.layer.classifier.in_features), 2)
		#self.nl = nn.LogSoftmax(dim = 1)
	
	def forward(self, X):
		x = self.layer(X)
		#x = self.nl(x)
		return x

model = Model()
#model.load_state_dict(torch.load('mls.pth'))


#model = torch.nn.DataParallel(model)#.cuda()

# Freezing all layers except classifier layers
for name, params in model.named_parameters():
	if 'denseblock4' in name or 'norm5' in name or 'classifier' in name or 'denseblock3' in name:
		params.requires_grad = True
	else:
		params.requires_grad = False


optimizer = torch.optim.Adam(
	model.parameters(),
	lr = 0.01, 
)

epochs = 400
minvalloss = float('inf')
trainlossrecord = [];vallossrecord = [];valaccrecord = [];

nlogloss = nn.NLLLoss()


for i in range(epochs):
    model.train()

    pbar = tqdm(total = trainsetsize)
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        images_all = torch.cat(images, 0)#.cuda()
        targets = targets#.cuda()
        logits_all = model(images_all)
        logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
        loss = F.cross_entropy(logits_clean, targets)
        p_clean, p_aug1, p_aug2 = F.softmax(
            logits_clean, dim = 1), F.softmax(
                logits_aug1, dim = 1), F.softmax(
                    logits_aug2, dim = 1)

        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2)/3., 1e-7, 1).log()
        loss+= 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') + F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

        loss.backward()
        optimizer.step()
        pbar.update(trainbatchsize//2)

    model.eval()
    trainloss = 0.
    for i, (images, targets) in enumerate(train_loader):
        images_all = torch.cat(images, 0)#.cuda()
        targets = targets#.cuda()
        logits_all = model(images_all)
        logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
        trainloss = F.cross_entropy(logits_clean, targets)
        p_clean, p_aug1, p_aug2 = F.softmax(
            logits_clean, dim = 1), F.softmax(
                logits_aug1, dim = 1), F.softmax(
                    logits_aug2, dim = 1)

        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2)/3., 1e-7, 1).log()
        trainloss+= 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        pbar.update(trainbatchsize//2)


    valloss = 0.
    tbar = tqdm(total = validationsetsize)
    for i, (images, targets) in enumerate(train_loader):
        images_all = torch.cat(images, 0)#.cuda()
        targets = targets#.cuda()
        logits_all = model(images_all)
        logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
        valloss = F.cross_entropy(logits_clean, targets)
        p_clean, p_aug1, p_aug2 = F.softmax(
            logits_clean, dim = 1), F.softmax(
                logits_aug1, dim = 1), F.softmax(
                    logits_aug2, dim = 1)

        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2)/3., 1e-7, 1).log()
        valloss+= 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        tbar.update(validationbatchsize//2)


    pbar.set_postfix(trainloss = trainloss)
    pbar.close()
















