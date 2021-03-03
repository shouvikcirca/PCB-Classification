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
os.environ["CUDA_VISIBLE_DEVICES"]="0" 


torch.manual_seed(123)
np.random.seed(0)

rootdir = 'dataset/BalancedTest/'
testsetsize = len(os.listdir(rootdir + 'True')) + len(os.listdir(rootdir + 'False'))
testbatchsize = 34

####################################  Calculating mean and std for normalization #############
preprocess_mean_std_calculation = transforms.Compose([
	transforms.Resize(size = (256,256), interpolation = 4),
	transforms.ToTensor()
])

f = 1 # No need to change this
rootdirtrain = 'dataset/6000samples/fold{}'.format(f)+'/'
trainsetsize = len(os.listdir(rootdirtrain + 'Train/True')) + len(os.listdir(rootdirtrain + 'Train/False'))

train_data = datasets.ImageFolder(
	rootdirtrain + 'Train',
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


preprocess = transforms.Compose([
	transforms.Resize(size = (256,256), interpolation = 4),
	transforms.ToTensor(),
	transforms.Normalize(trainmean, trainstd)
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
	





class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer = densenet201(pretrained = False)
		self.layer.classifier = nn.Linear(int(self.layer.classifier.in_features), 2)
		self.nl = nn.LogSoftmax(dim = 1)
	
	def forward(self, X):
		x = self.layer(X)
		x = self.nl(x)
		return x

model = Model()
model = torch.nn.DataParallel(model).cuda()
modelpath = 'cmpth_checkpoints/exp8/checkpoints/fold1/'
modelpath = modelpath + os.listdir(modelpath)[0]
sm = torch.load(modelpath)
model.load_state_dict(sm.state_dict())

model = model.eval()

testloss = 0.;
test_gt = torch.Tensor([]).int()
test_pred = torch.Tensor([]).int()
testlossrecord = []

tbar = tqdm(total = testsetsize)
for samples, labels in testloader:
	samples = samples.cuda()
	labels = labels.numpy().astype('int32')
	labels = torch.from_numpy(np.eye(2)[labels].astype('float32')).cuda()
		
	with torch.no_grad():	
		pred = model(samples)
		test_pred = torch.cat([test_pred,torch.argmax(pred, dim=1).int().cpu()])
		test_gt = torch.cat([test_gt,torch.argmax(labels, dim = 1).int().cpu()])
		testloss+=compound_dice_loss(pred, labels, 'cpu')
		
	samples = samples.cpu();del samples;
	pred = pred.cpu();del pred;
	labels = labels.cpu();del labels;
	tbar.update(testbatchsize)
	

#testlossrecord.append(testloss.item())	
tp,fp,tn,fn = getMetrics(test_gt.int(),test_pred.int(), 0 ,1)
testacc = (tp+tn)/(tp+fp+tn+fn)
ndp = tp/(tp+fp) if (tp+fp)>0. else 'Undefined'
ndr = tp/(tp+fn) if (tp+fn)>0. else 'Undefined'
tp,fp,tn,fn = getMetrics(test_gt.int(),test_pred.int(), 1 ,0)
dp = tp/(tp+fp) if (tp+fp)>0. else 'Undefined'
dr = tp/(tp+fn) if (tp+fn)>0. else 'Undefined'

tbar.set_postfix( test_acc = testacc, NonDefectivePrecision = ndp, NonDefectiveRecall = ndr, DefectivePrecision = dp, DefectiveRecall = dr)
tbar.close()
#testaccrecord.append(testacc)	

"""		
with open('cmpth_checkpoints/logs/trainlossrecord.txt','w') as f:
	f.write(str(trainlossrecord))

with open('cmpth_checkpoints/logs/vallossrecord.txt','w') as f:
	f.write(str(vallossrecord))

with open('cmpth_checkpoints/logs/valaccrecord.txt','w') as f:
	f.write(str(valaccrecord))
"""



