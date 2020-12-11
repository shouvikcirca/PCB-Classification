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


torch.manual_seed(123)
np.random.seed(0)

rootdir = 'dataset/6000samples/fold1/'

trainsetsize = len(os.listdir(rootdir + 'Train/True')) + len(os.listdir(rootdir + 'Train/False'))
validationsetsize = len(os.listdir(rootdir + 'Validation/True')) + len(os.listdir(rootdir + 'Validation/False'))

trainbatchsize = 34
validationbatchsize = 34#validationsetsize


preprocess = transforms.Compose([
	transforms.Resize(size = (256,256), interpolation = 4),
	transforms.ToTensor()
])


train_data = datasets.ImageFolder(
	rootdir + 'Train',
	transform = preprocess
)

validation_data = datasets.ImageFolder(
	rootdir + 'Validation',
	transform = preprocess
)


trainloader = torch.utils.data.DataLoader(
	train_data,
	batch_size = trainbatchsize,
	shuffle = True,
)



validationloader = torch.utils.data.DataLoader(
	validation_data,
	batch_size = validationbatchsize,
)


def rand_bbox(size, lam):
	W = size[2]
	H = size[3]
	cut_rat = np.sqrt(1. - lam)
	cut_w = np.int(W * cut_rat)
	cut_h = np.int(H * cut_rat)
	cx = np.random.randint(W)
	cy = np.random.randint(H)
	bbx1 = np.clip(cx - cut_w // 2, 0 ,W)
	bbx2 = np.clip(cx - cut_h // 2, 0 ,H)
	bbx3 = np.clip(cx - cut_w // 2, 0 ,W)
	bbx4 = np.clip(cx - cut_h // 2, 0 ,H)
	return bbx1, bbx2, bbx3, bbx4
		

def compound_dice_loss(ypred, ytrue, device):
	a = ypred * ytrue
	b = 2.*a+1.
	c = ypred - ytrue
	d = (2.*c*c)+1.
	e = b+d
	f = ypred+ytrue+1.
	g = 1. -  e/f
	g = g.to(device)

	ytrueargmax = torch.argmax(ytrue, dim=1).float()
	trues = (ytrueargmax == 1.).sum().int().item()
	falses = (ytrueargmax == 0.).sum().int().item()

	if trues == 0:
		falseratio = 1;trueratio = 0;
	elif falses == 0:
		falseratio = 0.;trueratio = 1.;
	else:
		falseratio = falses/trues; trueratio = trues/falses;

	h = torch.Tensor([falseratio, trueratio]).float().to(device)
	i = g*h
	j = i.sum()
	return j
	


def save_checkpoint(state, filepath):
	torch.save(state, filepath)



model = densenet201(pretrained = False)
model.classifier = nn.Linear(int(model.classifier.in_features), 2)

optimizer = torch.optim.Adam(
	model.parameters(),
	lr = 0.01, 
)


epochs = 300
minvalloss = float('inf')
trainlossrecord = [];vallossrecord = [];valaccrecord = [];


for i in range(epochs):
	trainloss = 0.
	model = model.cuda()
	model.train()
	step_number = 1

	pbar = tqdm(total = trainsetsize)
	for samples, labels in trainloader:
		optimizer.zero_grad()
		labels = labels.numpy().astype('int32')
		labels = torch.from_numpy(np.eye(2)[labels].astype('float32')).cuda()
		samples = samples.cuda()
		pred = F.log_softmax(model(samples), dim = 1)
		trainloss = compound_dice_loss(pred, labels, 'cuda')
		trainloss.backward()
		optimizer.step()

		samples = samples.cpu();del samples;
		pred = pred.cpu();del pred;
		labels = labels.cpu();del labels;
		pbar.set_postfix(trainloss = trainloss.item())
		pbar.update(trainbatchsize)
	
	trainlossrecord.append(trainloss.item())
	model = model.cpu()
	model = model.eval()

	valacc = 0.;
	valloss = 0.;
	val_gt = torch.Tensor([]).int()
	val_pred = torch.Tensor([]).int()

	tbar = tqdm(total = validationsetsize)
	for samples, labels in validationloader:
		labels = labels.numpy().astype('int32')
		labels = torch.from_numpy(np.eye(2)[labels].astype('float32'))#.cuda()
		
		with torch.no_grad():	
			pred = F.log_softmax(model(samples), dim = 1)
		
			val_pred = torch.cat([val_pred,torch.argmax(pred, dim=1).int()])
			val_gt = torch.cat([val_gt,torch.argmax(labels, dim = 1).int()])

			valloss+=compound_dice_loss(pred, labels, 'cpu')
		
		samples = samples.cpu();del samples;
		pred = pred.cpu();del pred;
		labels = labels.cpu();del labels;
		tbar.update(validationbatchsize)


	vallossrecord.append(valloss.item())
	tbar.close()
	tp,fp,tn,fn = getMetrics(val_gt.int(),val_pred.int(), 0 ,1)
	valacc = (tp+tn)/(tp+fp+tn+fn)
	valaccrecord.append(valacc)	
	pbar.set_postfix(epoch = i, train_loss = trainloss.item(), val_loss = valloss.item(), val_acc = valacc)
	pbar.close()

	if valloss < minvalloss:
		if len(os.listdir('cmpth_checkpoints/checkpoints')) > 0:
			os.system('rm cmpth_checkpoints/checkpoints/*')
		savepath = datetime.now().strftime("%d_%m_%Y___%H_%M_%S")+'.pth.tar'
		save_checkpoint({
			'epoch':i,
			'state_dict':model.state_dict(),
			'optimizer':optimizer.state_dict()
		}, 'cmpth_checkpoints/checkpoints/'+savepath)
			

with open('cmpth_checkpoints/logs/trainlossrecord.txt','w') as f:
	f.write(str(trainlossrecord))

with open('cmpth_checkpoints/logs/vallossrecord.txt','w') as f:
	f.write(str(vallossrecord))

with open('cmpth_checkpoints/logs/valaccrecord.txt','w') as f:
	f.write(str(valaccrecord))



