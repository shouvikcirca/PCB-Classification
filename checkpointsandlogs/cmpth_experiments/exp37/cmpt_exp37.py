import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p")
parser.add_argument("-f")
args = parser.parse_args()

from PIL import Image as pil_image
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d
import time
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.p
from cmatrix import getMetrics
from tqdm import tqdm
import math
from datetime import datetime
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(123)
np.random.seed(0)

f = int(args.f)


rootdir = 'dataset/jeny_dataset/fold{}'.format(f)+'/'

trainsetsize = len(os.listdir(rootdir + 'Train/True')) + len(os.listdir(rootdir + 'Train/False'))
validationsetsize = len(os.listdir(rootdir + 'Validation/True')) + len(os.listdir(rootdir + 'Validation/False'))

trainbatchsize = 54
validationbatchsize = 54

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

preprocess = transforms.Compose([
	transforms.Resize(size = (256,256), interpolation = 4),
	transforms.ToTensor(),
	transforms.Normalize(trainmean, trainstd)
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
		


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.auxlayer = nn.Sequential(*list(resnext50_32x4d(pretrained = False).children())[:-3])
		self.mainlayer = nn.Sequential(*list(resnext50_32x4d(pretrained = False).children())[-3:-1])
		
		self.fc = nn.Linear(2048, 2, bias = False)
		self.auxpool = nn.AvgPool2d(kernel_size = 8, stride = 3)
		self.auxconv = nn.Conv2d(in_channels = 1024, out_channels = 128, kernel_size = 1, stride = 1)
		self.auxrelu = nn.ReLU()
		self.bn = nn.BatchNorm2d(num_features  = 128)
		self.auxpool2 = nn.AvgPool2d(kernel_size = 4)
		self.auxfc = nn.Linear(128,64)
		self.auxfc2 = nn.Linear(64,2)
		self.finalLinear = nn.Linear(2048,2)
		#rn = abs(torch.randn(()))
		#self.pt = nn.Parameter(rn)
		#self.nl = nn.LogSoftmax(dim = 1)
	
	def forward(self, X):
		x = self.auxlayer(X)
		y = self.mainlayer(x)
		y = y.view(y.shape[0], y.shape[1])
		y = self.finalLinear(y)
		#########
		z = self.auxpool(x)
		z = self.auxconv(z)
		z = self.bn(z)
		z = self.auxrelu(z)
		z = self.auxpool2(z)
		z = z.view(z.shape[0], z.shape[1])
		z = nn.ReLU()(self.auxfc(z))
		z = self.auxfc2(z)
		###
		#y = self.pt * y
		#y = self.nl(y)
		return z,y

model = Model()

model = model.cuda()
model = torch.nn.DataParallel(model).cuda()

# Freezing all layers except classifier layers
#for name, params in model.named_parameters():
#	if 'denseblock4' in name or 'norm5' in name or 'classifier' in name or 'denseblock3' in name or 'transition3' in name:
#		params.requires_grad = True
#	else:
#		params.requires_grad = False


optimizer = torch.optim.Adam(
	model.parameters(),
	lr = 0.0001, 
)

#scheduler = StepLR(optimizer, step_size = 75, gamma = 0.1)


epochs = 1000
minvalloss = float('inf')
trainlossrecord = [];vallossrecord = [];valaccrecord = [];


celoss = nn.CrossEntropyLoss()
#nlogloss = nn.NLLLoss()

"""
for name, parameters in model.named_parameters():
	print(name)

"""
"""
dummydata = torch.rand([5,3,256,256]).cuda()
p = model(dummydata)
print(p[0].shape, p[1].shape)
"""



"""
for samples, labels in trainloader:
	samples = samples.cuda()
	pred = model(samples)
	#print(pred.shape)
	print(pred[0].shape, pred[1].shape)
	samples = samples.cpu()
	del samples
"""


for i in range(epochs):
	#model = model.cuda()
	model.train()
	#scheduler.step()

	pbar = tqdm(total = trainsetsize)
	for samples, labels in trainloader:
		optimizer.zero_grad()
		#labels = labels.numpy().astype('int32')
		#labels = torch.from_numpy(np.eye(2)[labels].astype('float32')).cuda()
		r = np.random.rand(1)
		if r<0.6:
			lam = 0.5
			rand_index = np.random.permutation(samples.shape[0])
			bbx1, bby1, bbx2, bby2 = rand_bbox(samples.shape, lam)
			samples = samples.numpy()
			samples[:,:,bbx1:bbx2,bby1:bby2] = samples[rand_index,:,bbx1:bbx2,bby1:bby2]
			samples = torch.from_numpy(samples)

		samples = samples.cuda()
		labels = labels.long().cuda()
		pred = model(samples)

		auxloss = celoss(pred[0], labels)
		auxloss.backward()

		optimizer.step()
		
		pbar.update(trainbatchsize//4)
		optimizer.zero_grad()
		pred = model(samples)
		mainloss = celoss(pred[1], labels)
		mainloss.backward()
		optimizer.step()	
		
		samples = samples.cpu();del samples;
		#pred = pred[0].cpu();del pred;
		labels = labels.cpu();del labels;

		pbar.update(trainbatchsize//4)


	model.eval()

	totaltrainloss = 0.
	for samples, labels in trainloader:
		samples = samples.float().cuda()
		labels = labels.long().cuda()
	
		with torch.no_grad():
			pred = model(samples)[1]
			totaltrainloss+=celoss(pred, labels)
			
		
		samples = samples.cpu();del samples;
		#pred = pred.cpu();del pred;
		labels = labels.cpu();del labels;
		pbar.update(trainbatchsize//2)
		

	pbar.set_postfix(trainloss = totaltrainloss.item())	
	trainlossrecord.append(totaltrainloss.item())

	valacc = 0.;
	valloss = 0.;
	val_gt = torch.Tensor([]).int()
	val_pred = torch.Tensor([]).int()

	tbar = tqdm(total = validationsetsize)
	for samples, labels in validationloader:
		samples = samples.float().cuda()
		labels = labels.long().cuda()		

		with torch.no_grad():	
			pred = model(samples)[1]
		
			val_pred = torch.cat([val_pred,torch.argmax(pred, dim=1).int().cpu()])
			val_gt = torch.cat([val_gt, labels.int().cpu()])

			valloss+=celoss(pred, labels)
		
		samples = samples.cpu();del samples;
		#pred = pred.cpu();del pred;
		labels = labels.cpu();del labels;
		tbar.update(validationbatchsize)


	vallossrecord.append(valloss.item())
	tbar.close()
	tp,fp,tn,fn = getMetrics(val_gt.int(),val_pred.int(), 0 ,1)
	valacc = (tp+tn)/(tp+fp+tn+fn)
	valaccrecord.append(valacc)	
	pbar.set_postfix(epoch = i, train_loss = totaltrainloss.item(), val_loss = valloss.item(), val_acc = valacc)
	pbar.close()
	if valloss < minvalloss:
		if len(os.listdir('cmpth_checkpoints/exp37/checkpoints/fold{}'.format(f))) > 0:
			os.system('rm cmpth_checkpoints/exp37/checkpoints/fold{}/*'.format(f))
		savepath = 'cmpth_checkpoints/exp37/checkpoints/fold{}/'.format(f)+ 'fold{}_'.format(f) + datetime.now().strftime("%d_%m_%Y___%H_%M_%S")+'.pth'
		torch.save(model, savepath)
			


wpath = 'cmpth_checkpoints/exp37/logs/fold{}/'.format(f)+'trainlossrecord.npy'
trainlossrecord = np.array(trainlossrecord)
np.save(wpath, trainlossrecord)

wpath = 'cmpth_checkpoints/exp37/logs/fold{}/'.format(f)+'vallossrecord.npy'
vallossrecord = np.array(vallossrecord)
np.save(wpath, vallossrecord)

wpath = 'cmpth_checkpoints/exp37/logs/fold{}/'.format(f)+'valaccrecord.npy'
valaccrecord = np.array(valaccrecord)
np.save(wpath, valaccrecord)

