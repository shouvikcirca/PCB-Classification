import warnings
warnings.filterwarnings('ignore')
from keras.preprocessing.image import ImageDataGenerator
from time import time
import os
import math
from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow as tf
import keras
from keras.applications import DenseNet121
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras import models, layers, optimizers
print('Imported libraries')



traincount = len(os.listdir('dataset/fold1/train/true')) + len(os.listdir('dataset/fold1/train/false'))
valcount = len(os.listdir('dataset/fold1/validation/true')) + len(os.listdir('dataset/fold1/validation/false'))

trainbatchsize = 32
valbatchsize = 32
print('Retrieved datacount')


train_datagen = ImageDataGenerator(rescale=1/255.)
train_generator = train_datagen.flow_from_directory(
        'dataset/fold1/train',
        batch_size=trainbatchsize,
        class_mode = 'binary'
)
print('Created Train Generator')


valid_datagen = ImageDataGenerator(rescale = 1/255.)
valid_generator = valid_datagen.flow_from_directory(
        'dataset/fold1/validation',
        batch_size = valbatchsize,
        class_mode = 'binary'
)
print('Created Validation Generator')


from Models.models import dn121scratch as dns
print('Imported Densenet')


import keras.backend as K
K.clear_session()
if 'model' in dir():
    del model
print('Deleted possible existing computation graphs')


model = dns()
print('Instantiated Model')


from datetime import datetime
f = 1


logdir = "logs/"+datetime.now().strftime("%d_%m_%Y____%H_%M_%S")
tboard = TensorBoard(log_dir = logdir)
print('Instantiated Tensorboard')
print(logdir)


model.compile(optimizer = optimizers.Adam(lr = 1e-3),loss = 'binary_crossentropy',metrics = ['accuracy'])
print('Compiled Model')


file_path = 'checkpoints1/'+'fold_{}____'.format(f)+datetime.now().strftime("%d_%m_%Y____%H_%M_%S")+'.h5'
my_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath = file_path,
                monitor = 'val_loss',
                save_best_only = True
            ),
            tboard
]
print('Defined Callback')
print('Checkpoint: '+ file_path)



print('Starting training')
history = model.fit(
    train_generator,
    steps_per_epoch = math.ceil(traincount/trainbatchsize),
    epochs=30,
    validation_data = valid_generator,
    validation_steps= math.ceil(valcount/valbatchsize),
    callbacks = my_callbacks
)

















