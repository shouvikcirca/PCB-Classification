'''
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow as tf
import keras  
from keras.applications import DenseNet121
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras import models, layers, optimizers
from tensorflow.keras import Input
from tensorflow.keras import layers, Model
'''

# Importing Model
'''
from Models.models import densenet
model = densenet(201)
model.compile(optimizer = optimizers.Adam(lr = 1e-3),loss = 'binary_crossentropy',metrics = ['accuracy'])
'''
'''
import os
traintruedirpath = 'Folds/3folds/fold1/Train/True'
trues = os.listdir(traintruedirpath)
xtraintrue = np.zeros([len(trues),600,600,3])
'''





import io
from PIL import Image as pil_image


'''
img = ''
with open('10.tif', 'rb') as f:
        img = pil_image.open(io.BytesIO(f.read()))
img = img.resize((256,256),pil_image.NEAREST)

img.show()

from tensorflow.keras.preprocessing.image import img_to_array
imgnp = img_to_array(img)
print(imgnp.shape)


#print(img.mode)
'''



'''
import os 
import numpy as np
import math
from keras.preprocessing.image import load_img, img_to_array, array_to_img


ims = np.zeros([17,600,600,3])
files = os.listdir()
counter = 0
for i in files:
    if 'tif' in i:
        ims[counter] = img_to_array(load_img(i))
        counter+=1


lam = np.random.uniform(low = 0, high = 1)
W = 600
H = 600
cut_rat = np.sqrt(1. - lam)
cut_w = np.int(W * cut_rat)
cut_h = np.int(H * cut_rat)

# uniform
cx = np.random.randint(W)
cy = np.random.randint(H)

bbx1 = np.clip(cx - cut_w , 0, W)
bby1 = np.clip(cy - cut_h , 0, H)
bbx2 = np.clip(cx + cut_w , 0, W)
bby2 = np.clip(cy + cut_h , 0, H)

array_to_img(ims[2]).show()
#array_to_img(ims[5]).show()
ims[2, :, bbx1:bbx2, bby1:bby2] = ims[5, :, bbx1:bbx2, bby1:bby2]


array_to_img(ims[2]).show()

'''

'''
patchx = math.floor(np.random.uniform()*600)
patchy = math.floor(np.random.uniform()*600)
l = np.random.uniform(low = 0, high = 1)
patchwidth = 600*np.sqrt(1 - l)
patchheight = 600*np.sqrt(1 - l)

print(patchx, patchy,patchheight, patchwidth)
'''

