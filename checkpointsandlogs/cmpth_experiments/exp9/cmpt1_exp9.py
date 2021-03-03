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

torch.manual_seed(123)
np.random.seed(0)

f = 3

rootdir = 'dataset/6000samples/fold{}'.format(f)+'/'

trainsetsize = len(os.listdir(rootdir + 'Train/True')) + len(os.listdir(rootdir + 'Train/False'))
validationsetsize = len(os.listdir(rootdir + 'Validation/True')) + len(os.listdir(rootdir + 'Validation/False'))

trainbatchsize = 64
validationbatchsize = 34

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
		

def compound_dice_loss(ypred, ytrue, device):
	a = ypred * ytrue
	b = 2.*a+1.
	c = ypred - ytrue
	d = (2.*c*c)+1.
	e = b+d
	f = ypred+ytrue+1.
	g = 1. -  e/f
	t = g.mean(dim = 0)
	t = t.to(device)
	j = t.sum()
	return j
	


def save_checkpoint(state, filepath):
	torch.save(state, filepath)


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
model.load_state_dict(torch.load('mls.pth'))


model = torch.nn.DataParallel(model).cuda()

# Freezing all layers except classifier layers
for name, params in model.named_parameters():
	if 'denseblock4' in name or 'norm5' in name or 'classifier' in name or 'denseblock3' in name or 'transition3' in name:
		params.requires_grad = True
	else:
		params.requires_grad = False


optimizer = torch.optim.SGD(
	model.parameters(),
	lr = 0.1,
	momentum = 0.9,
	weight_decay = 1e-4,
	nesterov = True 
)



#scheduler = StepLR(optimizer, step_size = 75, gamma = 0.1)
def lratescheduler(optimizer, epochs, epoch):
	if epoch == (0.5*epochs) or epoch == (0.75*epochs):
		for param_group in optimizer.param_groups:
			param_group['lr'] = 0.1*param_group['lr']
	return optimizer
	

epochs = 360
minvalloss = float('inf')
trainlossrecord = [];vallossrecord = [];valaccrecord = [];


#celoss = nn.CrossEntropyLoss()
nlogloss = nn.NLLLoss()

for i in range(epochs):
	model.train()
	#scheduler.step()
	optimizer = lratescheduler(optimizer, epochs, i)

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
		trainloss = nlogloss(pred, labels)
		trainloss.backward()
		optimizer.step()
		
		samples = samples.cpu();del samples;
		pred = pred.cpu();del pred;
		labels = labels.cpu();del labels;
		if 'truesamples' in dir():
			truesamples = truesamples.cpu();del truesamples;
			truelabels = truelabels.cpu();del truelabels;
		
		if 'falsesamples' in dir():
			falsesamples = falsesamples.cpu();del falsesamples;
			falselabels = falselabels.cpu();del falselabels;

		pbar.update(trainbatchsize//2)


	model = model.eval()

	totaltrainloss = 0.
	for samples, labels in trainloader:
		samples = samples.float().cuda()
		labels = labels.long().cuda()
	
		with torch.no_grad():
			pred = model(samples)
			totaltrainloss+=nlogloss(pred, labels)
		
		samples = samples.cpu();del samples;
		pred = pred.cpu();del pred;
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
		#labels = labels.numpy().astype('int32')
		#labels = torch.from_numpy(np.eye(2)[labels].astype('float32')).cuda()
		labels = labels.long().cuda()		

		with torch.no_grad():	
			pred = model(samples)
		
			val_pred = torch.cat([val_pred,torch.argmax(pred, dim=1).int().cpu()])
			#val_gt = torch.cat([val_gt,torch.argmax(labels, dim=1).int().cpu()])
			val_gt = torch.cat([val_gt, labels.int().cpu()])

			#valloss+=compound_dice_loss(pred, labels, 'cpu')
			valloss+=nlogloss(pred, labels)
		
		samples = samples.cpu();del samples;
		pred = pred.cpu();del pred;
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
		if len(os.listdir('cmpth_checkpoints/exp9/checkpoints/fold{}'.format(f))) > 0:
			os.system('rm cmpth_checkpoints/exp9/checkpoints/fold{}/*'.format(f))
		savepath = 'cmpth_checkpoints/exp9/checkpoints/fold{}/'.format(f)+ 'fold{}_'.format(f) + datetime.now().strftime("%d_%m_%Y___%H_%M_%S")+'.pth'
		torch.save(model, savepath)
			


wpath = 'cmpth_checkpoints/exp9/logs/fold{}/'.format(f)+'trainlossrecord.npy'
trainlossrecord = np.array(trainlossrecord)
np.save(wpath, trainlossrecord)

wpath = 'cmpth_checkpoints/exp9/logs/fold{}/'.format(f)+'vallossrecord.npy'
vallossrecord = np.array(vallossrecord)
np.save(wpath, vallossrecord)

wpath = 'cmpth_checkpoints/exp9/logs/fold{}/'.format(f)+'valaccrecord.npy'
valaccrecord = np.array(valaccrecord)
np.save(wpath, valaccrecord)
