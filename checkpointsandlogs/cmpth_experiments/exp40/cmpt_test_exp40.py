
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p")
parser.add_argument("-f")
args = parser.parse_args()





from lsoftmax import LSoftmaxLinear
from PIL import Image as pil_image
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d    #densenet121 #resnext50_32x4d
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
testsetsize = len(os.listdir(rootdir + 'Validation/True')) + len(os.listdir(rootdir + 'Validation/False'))

trainbatchsize = 64
testbatchsize = 64

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



test_data = datasets.ImageFolder(
	'dataset/jeny_dataset/Test',
	transform = preprocess
)


testloader = torch.utils.data.DataLoader(
	test_data,
	batch_size = testbatchsize,
	shuffle = True,
)



class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.layer = nn.Sequential(*list(resnext50_32x4d(pretrained = False).children())[:-2])
		self.fc = nn.Linear(2048, 2, bias = False)
		rn = torch.randn(())
		self.pt = nn.Parameter(rn)
	
	def forward(self, X):
		x = self.layer(X)
		x = nn.AvgPool2d(8)(x)
		x = x.view(x.shape[0],x.shape[1])
		y = (x*x).detach()
		distribution_sum = (y.sum(dim=1)**0.5).unsqueeze(1).detach()
		x = x/distribution_sum
		x = self.pt * x
		x = self.fc(x)
		return x

model = Model()
model = model.cuda()
#model = torch.nn.DataParallel(model).cuda()
modelpath = 'cmpth_checkpoints/exp40/checkpoints/fold{}'.format(str(f))
modelcp = modelpath+'/'+os.listdir(modelpath)[0]
sm = torch.load(modelcp)
model.load_state_dict(sm.state_dict())

model.eval()

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
		#testloss+=compound_dice_loss(pred, labels, 'cpu')
		
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
