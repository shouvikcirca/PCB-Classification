import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import time
#from keras.callbacks import TensorBoard
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import tensorflow
#import keras  
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from tensorflow.keras import models, layers, optimizers
from Models.models import densenetforaugmix as dns
import tensorflow.keras.backend as K
import math
from datetime import datetime
from PIL import Image, ImageOps, ImageEnhance
import augmentations
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
from tensorflow.keras.utils import to_categorical
from cmatrix import getMetrics


np.random.seed(123)
rootdir = 'Folds/'


################## AugMentations ################################
def aug(imagearray):
    """Perform AugMix augmentations and compute mixture.
    Args:
        image: PIL.Image input image
        preprocess: Preprocessing function which should return a torch tensor.
    Returns:
        mixed: Augmented and mixed image.
    """
    width = np.random.randint(1,10)
    depth = np.random.randint(1,10)
    aug_severity = np.random.uniform(0,1)

    aug_list = augmentations.augmentations

    ws = np.float32(np.random.dirichlet([1] * width))
    m = np.float32(np.random.beta(1, 1))

    for imind in range(imagearray.shape[0]):
        image = imagearray[imind]
        mix = np.zeros_like(image)
        for i in range(width):
            image_aug = image.copy()
            depth = depth if depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                image_aug = img_to_array(op(array_to_img(image_aug), aug_severity))
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * image_aug
        mixed = (1 - m) * image + m * mix
        imagearray[imind] = mixed

    return imagearray

#########################################################

train_datagen = ImageDataGenerator(rescale = 1/255.)
valid_datagen = ImageDataGenerator(rescale = 1/255.)


validationdatasize = len(os.listdir(rootdir + 'fold1/Validation/False'))+ len(os.listdir(rootdir + 'fold1/Validation/True'))
traindatasize = len(os.listdir(rootdir + 'fold1/Train/False'))+ len(os.listdir(rootdir + 'fold1/Train/True'))

trainbatchsize = 2
validationbatchsize = validationdatasize

train_generator = train_datagen.flow_from_directory(
	rootdir + 'fold1/Train',
    	batch_size = trainbatchsize,
    	class_mode = 'binary',
    	shuffle = True,
    	seed = 123,
	target_size = (32,32),
	color_mode = 'rgb',
        interpolation = 'bicubic'
)

valid_generator = valid_datagen.flow_from_directory(
	rootdir + 'fold1/Validation',
    	batch_size = validationbatchsize,
    	class_mode = 'binary',
    	shuffle = True,
    	seed = 123,
	target_size = (32,32),
	color_mode = 'rgb',
        interpolation = 'bicubic'
)

K.clear_session()
if 'model' in dir():
    del model

f = 1
cpointsavepath = 'Foldscpandlogs/checkpoints/'+'fold_{}____'.format(f)+datetime.now().strftime("%d_%m_%Y____%H_%M_%S")+'.h5'

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


model = dns(121)
model.compile(metrics = ['accuracy'])
optimizer = tensorflow.keras.optimizers.Adam(learning_rate=1e-3)

epochs = 10
for epoch in range(epochs):
    for images, labels in train_generator:
        loss = 0.
        labels = to_categorical(labels.astype('int32'), num_classes = 2)
        aug1 = aug(images)
        aug2 = aug(images)

        with tensorflow.GradientTape() as tape:
            pred  = model(images)
            pred1 = model(aug1)
            pred2 = model(aug2)
        
            loss = tensorflow.nn.softmax_cross_entropy_with_logits(
                labels = tensorflow.stop_gradient(labels), logits = pred, axis = 1
            )

            p_clean = tensorflow.nn.softmax(pred, axis = 1)
            p_aug1 = tensorflow.nn.softmax(pred1, axis = 1)
            p_aug2 = tensorflow.nn.softmax(pred2, axis = 1)
        
            p_mixture = tensorflow.math.log( tensorflow.clip_by_value( 
                (p_clean+p_aug1+p_aug2)/3., 
                clip_value_min=1e-7, 
                clip_value_max=1)
            )

            loss = loss + (\
                    tensorflow.math.reduce_mean(tensorflow.keras.losses.KLDivergence()(p_clean, p_mixture)) + \
                    tensorflow.math.reduce_mean(tensorflow.keras.losses.KLDivergence()(p_aug1, p_mixture)) + \
                    tensorflow.math.reduce_mean(tensorflow.keras.losses.KLDivergence()(p_aug2, p_mixture)) \
            )/3.
    
            loss = tensorflow.math.reduce_mean(loss)
     
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
   
    for samples, labels in valid_generator:
        print(model.evaluate(samples,labels))
        #preds = np.argmax(model(samples), axis = 1).astype('float32')
        #tp,fp,tn,fn = getMetrics(labels, preds, 0, 1)

        #print('Validation Accuracy:{}'.format((tp+tn)/(tp+tn+fp+fn)))
    





   
    







    #with tf.GradientTape() as tape:
        







"""
history = model.fit_generator(
    train_generator,
    epochs = 2
    steps_per_epoch = math.ceil(traindatasize/trainbatchsize),
    validation_data = valid_generator,
    validation_steps = math.ceil(validationdatasize/validationbatchsize),
    callbacks = my_callbacks,    verbose = 1,
)
"""








with open('Foldstrainlog.txt','a') as f:
	f.write('Fold 1 done\n')
























