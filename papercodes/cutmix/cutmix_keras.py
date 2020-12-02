import tensorflow
import numpy
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import os
import numpy as np


partition = {}
labels = {}

train = []
val = []

truepath = 'randimages/Train/True/'
trues = os.listdir(truepath)
for i in trues:
    train.append(truepath + i)
    labels[truepath + i] = 1
falsepath = 'randimages/Train/False/'
falses = os.listdir(falsepath)
for i in falses:
    train.append(falsepath + i)
    labels[falsepath + i] = 0

partition['train'] = train


truepath = 'randimages/Validation/True/'
trues = os.listdir(truepath)
for i in trues:
    val.append(truepath + i)
    labels[truepath + i] = 1
falsepath = 'randimages/Validation/False/'
falses = os.listdir(falsepath)
for i in falses:
    val.append(falsepath + i)
    labels[falsepath + i] = 0

partition['val'] = val


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(300,300), n_channels=3,n_classes=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[1]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) 

    def __data_generation(self, list_IDs_temp):
        X = np.empty((len(list_IDs_temp), *self.dim, self.n_channels))
        y = np.empty((len(list_IDs_temp)), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            X[i] = img_to_array(load_img(ID))
            y[i] = self.labels[ID]

        r = np.random.rand(1)
        print(r)
        if r<0.5:
            lam = 0.5
            rand_index = np.random.permutation(X.shape[0])
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(X.shape, lam)
            X[:,:,bbx1:bbx2,bby1:bby2] = X[rand_index,:,bbx1:bbx2,bby1:bby2]
            
            

        return X, y#tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)

    
    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index*self.batch_size) + min(   self.batch_size,   (len(self.list_IDs) - index * self.batch_size) )]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y


params = {'dim': (300,300),'batch_size': 12,'n_classes': 2,'n_channels': 3,'shuffle': True}
training_generator = DataGenerator(partition['train'], labels, **params)


iter = 1
for samples, targets in training_generator:
    array_to_img(samples[0]).show()
    inp = input()
















