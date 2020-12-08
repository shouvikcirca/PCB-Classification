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

torch.manual_seed(123)
np.random.seed(0)

rootdir = 'dataset/6000samples/fold1/'

trainsetsize = len(os.listdir(rootdir + 'Train/True')) + len(os.listdir(rootdir + 'Train/False'))
validationsetsize = len(os.listdir(rootdir + 'Validation/True')) + len(os.listdir(rootdir + 'Validation/False'))

trainbatchsize = 34
validationbatchsize = 34


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



def compound_dice_loss(ypred, ytrue):
	a = ypred * ytrue
	b = 2 * a.sum() + 1.
	c = b/(ypred.sum() + ytrue.sum() + 1.)
	return c


model = densenet201(pretrained = False)
model.classifier = nn.Linear(int(model.classifier.in_features), 2)
optimizer = torch.optim.Adam(
	model.parameters(),
	lr = 0.01, 
)


epochs = 2
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
		pred = model(samples)
		trainloss = compound_dice_loss(pred, labels)
		trainloss.backward()
		optimizer.step()

		samples = samples.cpu();del samples;
		pred = pred.cpu();del pred;
		labels = labels.cpu();del labels;
		pbar.set_postfix(trainloss = trainloss.item())
		pbar.update(trainbatchsize)
	
	model = model.cpu()
	model = model.eval()

	valacc = 0.;
	valloss = 0.;
	tps = 0.;
	fps = 0.;
	tns = 0.;
	fns = 0.;
	sij = 0.;
	pij = 0.;
	spsum = 0.;

	tbar = tqdm(total = validationsetsize)
	for samples, labels in validationloader:
		labels = labels.numpy().astype('int32')
		labels = torch.from_numpy(np.eye(2)[labels].astype('float32'))#.cuda()
		#samples = samples.cuda()
		pred = model(samples)

		spsum+=(pred*labels).sum()
		sij += labels.sum()
		pij += pred.sum()

		classpredictions = torch.argmax(pred, dim=1).int()
		tp,fp,tn,fn = getMetrics(torch.argmax(labels, dim = 1).int(), classpredictions, 0, 1)
		tps+=tp;fps+=fp;tns+=tn;fns+=fn;

		samples = samples.cpu();del samples;
		pred = pred.cpu();del pred;
		labels = labels.cpu();del labels;
		tbar.update(validationbatchsize)


	tbar.close()	
	valloss = (2. * spsum + 1.)/(sij + pij + 1.)
	valacc = (tps+fps)/(tps+fps+tns+fns)
	pbar.set_postfix(train_loss = trainloss.item(), val_loss = valloss.item(), val_acc = valacc)
	pbar.close()
	#print('Trainloss:{} Valloss:{} ValAcc:{}'.format(trainloss, valloss, valacc))
			
		


