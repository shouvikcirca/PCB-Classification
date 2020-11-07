import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  
from datetime import datetime
from keras import Input
from keras import layers, Model, optimizers
from keras.preprocessing.image import ImageDataGenerator
from time import time
import numpy as np
from keras.callbacks import ModelCheckpoint

input_img = Input(shape=(600, 600, 3))
x = layers.Conv2D(512, kernel_size = 3, strides = 2, activation='relu', padding='same')(input_img)
#----------
x = layers.Conv2D(256, kernel_size = 3, strides = 1, activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
#------------

x = layers.Conv2DTranspose(512, (3, 3), strides = 2, activation='relu', padding='same')(x)
#---------------
x = layers.Conv2DTranspose(3, (3, 3), strides = 2, activation='sigmoid', padding='same')(x)
#--------------
decoder = Model(input_img,x)
decoder.compile(optimizer = optimizers.Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1/255.)

train_generator = train_datagen.flow_from_directory(
	'dataset/fold3/train/false',
    	batch_size = 34,
    	class_mode = 'binary',
    	shuffle = True,
    	seed = 123,
	target_size = (600,600),
	color_mode = 'rgb'
)

f = 1

cpointsavepath = 'autoencodercheckpoints/'+'fold_{}____'.format(f)+datetime.now().strftime("%d_%m_%Y____%H_%M_%S")+'.h5'
mycallbacks = [
		ModelCheckpoint(
			filepath = cpointsavepath,
        		monitor = 'val_loss',
        		save_best_only = True
        	)
]




decoder.fit_generator(
	train_generator,
	epochs=20,
	steps_per_epoch = 21,
	callbacks = mycallbacks,
        shuffle=True,
        validation_data = train_generator,
	validation_steps = 21,
	verbose = 1,
	
)
