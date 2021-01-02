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
	mean, std = np.array(MEAN), np.array(STD)
	image = (image - mean[:, None, None])/std[:, None, None]
	return image.transpose(1, 2, 0) 


def apply_op(image, op, severity):
	image = np.clip(image * 255., 0., 255.).astype(np.uint8)
	pil_img = Image.fromarray(image)
	pil_img = op(pil_img)
	return np.asarray(pil_img)/255.


def aug(image, severity = 3, width = 3, depth = -1, alpha = 1.):
	ws = np.float32(np.random.dirichlet([alpha] * width))
	m = np.float32(np.random.beta(alpha, alpha))
	mix = np.zeros_like(image)
        auglist = augmentations.augmentations
	for i in range(width):
		image_aug = image.copy()
		d = depth if depth>0 else np.random.randint(1, 4)
		for _ in range(d):
			op = np.random.choice(auglist)
			image_aug = apply_op(image_aug, op, severity)
		mix+=ws[i] * normalize(image_aug)
	
	mixed  = (1 - m)*normalize(image) + m*mix
	return mixed


#########################################################


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
	transforms.Resize(size = (256,256), interpolation = 4),
	transforms.ToTensor(),
	transforms.Normalize(trainmean, trainstd)])

train_dataset = datasets.ImageFolder(
	rootdir + 'Train',
)

train_dataset = AugMixDataset(train_dataset, preprocess)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = trainbatchsize,
        shuffle = True
) 

validation_dataset = dataset.ImageFolder(
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
                print(type(images))
                print(type(targets))

                
		"""
                optimizer.zero_grad()
                images_all = torch.cat(images, 0).cuda()
                targets = targets.cuda()
                logits_all = model(images_all)
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                loss = F.cross_entropy(logits_clean, targets)
                
                p_clean, p_aug1, p_aug2 = F.softmax(
                        logits_clean, dim = 1), F.softmax(
                                logits_aug1, dim = 1), F.softmax(
                                        logits_aug2, dim = 1)

                p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2)/3., 1e-7, 1).log()
                loss+= 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

                loss.backward()
		optimizer.step()
                pbar.update(trainbatchsize)
                """
        """
	model = model.eval()
	#train_gt = torch.Tensor([]).int()
	#train_pred = torch.Tensor([]).int()

	totaltrainloss = 0.
	for samples, labels in trainloader:
		samples = samples.float().cuda()
		#labels = labels.numpy().astype('int32')
		#labels = torch.from_numpy(np.eye(2)[labels].astype('float32')).cuda()
		labels = labels.long().cuda()
	
		with torch.no_grad():
			pred = model(samples)
			#train_pred = torch.cat([train_pred, torch.argmax(pred, dim=1).int().cpu()])
			#train_gt = torch.cat([train_gt, torch.argmax(labels, dim=1).int().cpu()])
			totaltrainloss+=nlogloss(pred, labels)
			#totaltrainloss+=compound_dice_loss(pred, labels, 'cpu')
		
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
		if len(os.listdir('cmpth_checkpoints/exp7/checkpoints/fold{}'.format(f))) > 0:
			os.system('rm cmpth_checkpoints/exp7/checkpoints/fold{}/*'.format(f))
		savepath = 'cmpth_checkpoints/exp7/checkpoints/fold{}/'.format(f)+ 'fold{}_'.format(f) + datetime.now().strftime("%d_%m_%Y___%H_%M_%S")+'.pth'
		torch.save(model, savepath)
        """			

"""
wpath = 'cmpth_checkpoints/exp7/logs/fold{}/'.format(f)+'trainlossrecord.npy'
trainlossrecord = np.array(trainlossrecord)
np.save(wpath, trainlossrecord)

wpath = 'cmpth_checkpoints/exp7/logs/fold{}/'.format(f)+'vallossrecord.npy'
vallossrecord = np.array(vallossrecord)
np.save(wpath, vallossrecord)

wpath = 'cmpth_checkpoints/exp7/logs/fold{}/'.format(f)+'valaccrecord.npy'
valaccrecord = np.array(valaccrecord)
np.save(wpath, valaccrecord)
"""
