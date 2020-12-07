import os
from keras.preprocessing.image import ImageDataGenerator
from time import time
#from keras.callbacks import TensorBoard
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
import tensorflow as tf
import keras  
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras import models, layers, optimizers
from Models.models import densenetsoftmax as dns
import keras.backend as K
import math
from datetime import datetime
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical

np.random.seed(123)

################################################################
partition = {}
labels = {}

train = []
val = []

rootdir = 'dataset/6000samples/fold1/'

truepath = rootdir + '/Train/True/'
trues = os.listdir(truepath)
for i in trues:
    train.append(truepath + i)
    labels[truepath + i] = 1
falsepath = rootdir + 'Train/False/'
falses = os.listdir(falsepath)
for i in falses:
    train.append(falsepath + i)
    labels[falsepath + i] = 0

partition['train'] = train


truepath = rootdir + 'Validation/True/'
trues = os.listdir(truepath)
for i in trues:
    val.append(truepath + i)
    labels[truepath + i] = 1
falsepath = rootdir + 'Validation/False/'
falses = os.listdir(falsepath)
for i in falses:
    val.append(falsepath + i)
    labels[falsepath + i] = 0

partition['val'] = val


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(600,600), n_channels=3,n_classes=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def rand_bbox(self, size, m):
        W = size[2]
        H = size[1]
        cut_rat = np.sqrt(1. - m)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w //2, 0, W)
        bby1 = np.clip(cy - cut_h //2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) 

    def __data_generation(self, list_IDs_temp):
        X = np.empty((len(list_IDs_temp), 256, 256, 3))
        y = np.empty((len(list_IDs_temp)))

        for i, ID in enumerate(list_IDs_temp):
            X[i] = img_to_array(load_img(path = ID, target_size = (256,256), interpolation = 'bicubic'))
            y[i] = self.labels[ID]

        r = np.random.rand(1)
        if r<0.5:
            m = 0.5
            rand_index = np.random.permutation(X.shape[0])
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(X.shape, m)
            X[:,:,bbx1:bbx2,bby1:bby2] = X[rand_index,:,bbx1:bbx2,bby1:bby2]
            lam = 1 - (((bbx2 - bbx1)*(bby2 - bby1))/(self.dim[0]*self.dim[1]))
            y = (lam * y) + ((1 - lam)*  y[rand_index])
        
        X/=255.
        y = to_categorical(y)
        return X, y#tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)

    
    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index*self.batch_size) + min(   self.batch_size,   (len(self.list_IDs) - index * self.batch_size) )]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y



trainbatchsize = 2
params = {'dim': (600,600),'batch_size': trainbatchsize,'n_channels': 3,'shuffle': True}
training_generator = DataGenerator(partition['train'], labels, **params)


###############################################################

validationbatchsize = 464
valid_datagen = ImageDataGenerator(rescale = 1/255.)


traindatasize = len(os.listdir(rootdir + 'Train/False'))+ len(os.listdir(rootdir + 'Train/True'))
validationdatasize = len(os.listdir(rootdir + 'Validation/False'))+ len(os.listdir(rootdir + 'Validation/True'))

valid_generator = valid_datagen.flow_from_directory(
	rootdir + 'Validation',
    	batch_size = validationbatchsize,
    	class_mode = 'binary',
    	shuffle = True,
    	seed = 123,
	target_size = (256,256),
	color_mode = 'rgb',
	interpolation = 'bicubic'
)

K.clear_session()
if 'model' in dir():
    del model

f = 1
cpointsavepath = 'cutmix_checkpointsandlogs/checkpoints/6000samples_400e_decay_strategy/'+'fold_{}____'.format(f)+datetime.now().strftime("%d_%m_%Y____%H_%M_%S")+'.h5'

#tboardfilepath = "augmented_checkpointsandlogs/logs/"+'Fold{}__'.format(f)+datetime.now().strftime("%d_%m_%Y____%H_%M_%S")
#tboard = TensorBoard(log_dir = tboardfilepath)
def scheduler(epoch, lr):
	if epoch in (75,150,225,300):
		return lr*0.1
	else:
		return lr


my_callbacks = [
            ModelCheckpoint(
                filepath = cpointsavepath,
                monitor = 'val_loss',
                save_best_only = True
            ),
            LearningRateScheduler(scheduler)
        ]


model = dns()

def custom_loss(ypred, ytrue):
    falses = (np.argmax(ytrue, axis = 1) == 0).sum()
    trues = (np.argmax(ytrue, axis = 1) == 1).sum()
    loss = K.square(ypred - ytrue)
    weights = [falses/trues, trues/falses]
    loss = loss * weights
    loss = K.sum(loss, axis = 1)
    return loss


model.compile(optimizer = optimizers.Adam(lr = 0.01),loss = custom_loss, ,metrics = ['accuracy'])
history = model.fit_generator(
    training_generator,
    epochs = 400,
    steps_per_epoch = math.ceil(traindatasize/trainbatchsize),
    validation_data = valid_generator,
    validation_steps = math.ceil(validationdatasize/validationbatchsize),
    callbacks = my_callbacks,
    verbose = 1,
)


fw = 'Fold '+str(f)+' done\n' 
with open('cutmixtrainlog.txt','a') as f:
	f.write(fw)
























