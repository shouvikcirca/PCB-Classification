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

rootdir = 'dataset/jeny_dataset/Test/'
testsetsize = len(os.listdir(rootdir + 'True')) + len(os.listdir(rootdir + 'False'))
testbatchsize = 34


preprocess = transforms.Compose([
	transforms.Resize(size = (256,256), interpolation = 4),
	transforms.ToTensor(),
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
modelpath = 'cmpth_checkpoints/exp12/checkpoints/fold1/'
modelpath = modelpath + os.listdir(modelpath)[0]
sm = torch.load(modelpath)
model.load_state_dict(sm.state_dict())

model = model.eval()

testloss = 0.;
test_gt = torch.Tensor([]).int()
test_pred = torch.Tensor([]).int()
testlossrecord = []


nlogloss = nn.NLLLoss()

tbar = tqdm(total = testsetsize)
for samples, labels in testloader:
	samples = samples.float().cuda()
	labels = labels.long().cuda()	
	
	with torch.no_grad():	
		pred = model(samples)
		test_pred = torch.cat([test_pred,torch.argmax(pred, dim=1).int().cpu()])
		test_gt = torch.cat([test_gt,labels.int().cpu()])
		testloss+=nlogloss(pred, labels)
		
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
df = (2*dp*dr)/(dp + dr)
ndf = (2*ndp*ndr)/(ndp + ndr)


tbar.set_postfix(df1 = df, ndf1 = ndf, test_acc = testacc, NonDefectivePrecision = ndp, NonDefectiveRecall = ndr, DefectivePrecision = dp, DefectiveRecall = dr)
tbar.close()
#testaccrecord.append(testacc)	




