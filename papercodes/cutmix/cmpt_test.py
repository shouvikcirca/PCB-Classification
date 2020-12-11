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

rootdir = 'dataset/Test/'

testsetsize = len(os.listdir(rootdir + 'True')) + len(os.listdir(rootdir + 'False'))

testbatchsize = 34


preprocess = transforms.Compose([
	transforms.Resize(size = (256,256), interpolation = 4),
	transforms.ToTensor()
])


test_data = datasets.ImageFolder(
	rootdir,
	transform = preprocess
)


testloader = torch.utils.data.DataLoader(
	test_data,
	batch_size = testbatchsize,
	shuffle = True,
)


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
	


model = densenet201(pretrained = False)
model.classifier = nn.Linear(int(model.classifier.in_features), 2)

modelpth = os.listdir('cmpth_checkpoints/checkpoints')
model.load_state_dict(torch.load('cmpth_checkpoints/checkpoints/'+modelpth[0]))


"""
optimizer = torch.optim.Adam(
	model.parameters(),
	lr = 0.01, 
)
"""


model = model.cpu()
model = model.eval()

testloss = 0.;
test_gt = torch.Tensor([]).int()
test_pred = torch.Tensor([]).int()

tbar = tqdm(total = testsetsize)
for samples, labels in testloader:
	labels = labels.numpy().astype('int32')
	labels = torch.from_numpy(np.eye(2)[labels].astype('float32'))#.cuda()
		
	with torch.no_grad():	
		pred = F.log_softmax(model(samples), dim = 1)
		test_pred = torch.cat([val_pred,torch.argmax(pred, dim=1).int()])
		test_gt = torch.cat([val_gt,torch.argmax(labels, dim = 1).int()])
		testloss+=compound_dice_loss(pred, labels, 'cpu')
		
	samples = samples.cpu();del samples;
	pred = pred.cpu();del pred;
	labels = labels.cpu();del labels;
	tbar.update(testbatchsize)
	
	testlossrecord.append(testloss.item())
	
tbar.set_postfix(test_loss = testloss.item(), test_acc = testacc)
tbar.close()
tp,fp,tn,fn = getMetrics(test_gt.int(),test_pred.int(), 0 ,1)
testacc = (tp+tn)/(tp+fp+tn+fn)
testaccrecord.append(testacc)	

			
with open('cmpth_checkpoints/logs/trainlossrecord.txt','w') as f:
	f.write(str(trainlossrecord))

with open('cmpth_checkpoints/logs/vallossrecord.txt','w') as f:
	f.write(str(vallossrecord))

with open('cmpth_checkpoints/logs/valaccrecord.txt','w') as f:
	f.write(str(valaccrecord))




