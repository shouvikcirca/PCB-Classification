import os
from keras.preprocessing.image import ImageDataGenerator
from time import time
#from keras.callbacks import TensorBoard
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import tensorflow as tf
import keras  
from keras.applications import DenseNet121
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras import models, layers, optimizers
from Models.models import dn121scratch as dns
import keras.backend as K
import math
from datetime import datetime


trainbatchsize = 34
validationbatchsize = 464

train_datagen = ImageDataGenerator(rescale = 1/255.)
valid_datagen = ImageDataGenerator(rescale = 1/255.)


traindatasize = len(os.listdir('dataset/downsampled/fold1/Train/False'))+ len(os.listdir('dataset/downsampled/fold1/Train/True'))
validationdatasize = len(os.listdir('dataset/downsampled/fold1/Validation/False'))+ len(os.listdir('dataset/downsampled/fold1/Validation/True'))

train_generator = train_datagen.flow_from_directory(
	'dataset/downsampled/fold1/Train',
    	batch_size = trainbatchsize,
    	class_mode = 'binary',
    	shuffle = True,
    	seed = 123,
	target_size = (256,256),
	color_mode = 'rgb'
)

valid_generator = valid_datagen.flow_from_directory(
	'dataset/downsampled/fold1/Validation',
    	batch_size = validationbatchsize,
    	class_mode = 'binary',
    	shuffle = True,
    	seed = 123,
	target_size = (256,256),
	color_mode = 'rgb'
)

K.clear_session()
if 'model' in dir():
    del model

f = 1
cpointsavepath = 'downsampled_checkpointsandlogs/checkpoints/'+'fold_{}____'.format(f)+datetime.now().strftime("%d_%m_%Y____%H_%M_%S")+'.h5'

#tboardfilepath = "augmented_checkpointsandlogs/logs/"+'Fold{}__'.format(f)+datetime.now().strftime("%d_%m_%Y____%H_%M_%S")
#tboard = TensorBoard(log_dir = tboardfilepath)

my_callbacks = [
            ModelCheckpoint(
                filepath = cpointsavepath,
                monitor = 'val_loss',
                save_best_only = True
            )#,
            #tboard
        ]



model = dns()
model.compile(optimizer = optimizers.Adam(lr = 1e-3),loss = 'binary_crossentropy',metrics = ['accuracy'])
history = model.fit_generator(
    train_generator,
    epochs = 100,
    steps_per_epoch = math.ceil(traindatasize/trainbatchsize),
    validation_data = valid_generator,
    validation_steps = math.ceil(validationdatasize/validationbatchsize),
    callbacks = my_callbacks,
    verbose = 1,
)


with open('downsampledtrainlog.txt','a') as f:
	f.write('Fold 1 done\n')

























