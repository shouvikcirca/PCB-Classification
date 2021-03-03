import os
import math
from sklearn.metrics import classification_report,confusion_matrix
import skimage.feature as sf
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras
from keras import layers
from keras import backend
from keras import models
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras import initializers
from keras.engine.topology import Layer
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adadelta
from PIL import Image
import itertools
from cmatrix import getMetrics

class BasicCall(Callback):

    def on_epoch_end(self, epoch, logs={}):
        data = self.validation_data
        labels = np.argmax(data[1], axis=1)
        model = Model(inputs=self.model.input, outputs=self.model.get_layer('side_out').output)
        output = model.predict([data[0], data[1]])
        visualize_basic(output, labels, epoch)
        return


class CenterLossCall(Callback):

    def __init__(self, lambda_centerloss):
        super().__init__()
        self.lambda_centerloss = lambda_centerloss

    def on_epoch_end(self, epoch, logs={}):
        data = self.validation_data
        labels = np.argmax(data[1], axis=1)
        model = Model(inputs=self.model.input[0], outputs=self.model.get_layer('side_out').output)
        output = model.predict(data[0])
        centers = self.model.get_layer('centerlosslayer').get_weights()[0]
        visualize(output, labels, epoch, centers, self.lambda_centerloss)
        return



class Alpha_Print(Callback):

    def on_epoch_end(self, epoch, logs={}):
        print('---')
        print(type(self.model.get_layer('side_out').get_weights()))
        print(len(self.model.get_layer('side_out').get_weights()))
        print(self.model.get_layer('side_out').get_weights())
        print('---')


def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    bn_axis = 3 #if backend.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.PReLU(alpha_initializer=initializers.Constant(value=0.25))(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x

def conv_block(x, growth_rate, name):
    bn_axis = 3 #if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.PReLU(alpha_initializer=initializers.Constant(value=0.25))(x1)
    x1 = layers.SeparableConv2D(4 * growth_rate, 1,
                       use_bias=False, 
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.PReLU(alpha_initializer=initializers.Constant(value=0.25))(x1)
    x2 = layers.SeparableConv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,dilation_rate=(2, 2),
                       name=name + '_2_conv')(x1)
    x3 = layers.SeparableConv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,dilation_rate=(4, 4),
                       name=name + '_3_conv')(x1)
    x4 = layers.SeparableConv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,dilation_rate=(8, 8),
                       name=name + '_4_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1,x2,x3,x4])
    return x



class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(2, 10),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):

        # x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
        print('centerloss layer x shape = ',np.asarray(x).shape)
        print('centerloss layer center = ',np.asarray(self.centers).shape)
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True) #/ K.dot(x[1], center_counts)
        return self.result # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def DenseNet():
    bn_axis = 3
    #blocks = [6, 12, 18, 24]
    blocks = [6,12,32,32]

    img_input=layers.Input((100,100,4),name='inputlayer')
    labels=layers.Input((2,))
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.PReLU(alpha_initializer=initializers.Constant(value=0.25))(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.PReLU(alpha_initializer=initializers.Constant(value=0.25))(x)
    x2=GlobalAveragePooling2D()(x)
    x3=Dropout(0.3)(x2)
    out1=Dense(10)(x3)
    out1 = layers.PReLU(alpha_initializer=initializers.Constant(value=0.25),name='side_out')(out1)
    out2=Dense(2,activation='softmax', name='outputlayer')(out1)
    side = CenterLossLayer(alpha=0.5, name='centerlosslayer')([out1, labels])
    model=Model(inputs=[img_input,labels],outputs=[out2,side])
    return model

#############
trainsetsize = len(os.listdir('dataset/6000samples/fold3/Train/True/')) + len(os.listdir('dataset/6000samples/fold3/Train/False/'))
testsetsize = len(os.listdir('dataset/BalancedTest/True/')) + len(os.listdir('dataset/BalancedTest/False/'))
testbatchsize = testsetsize


partition = {}
labels = {}
train = []
test = []

truepath = 'dataset/6000samples/fold3/Train/True/'
trues = os.listdir(truepath)
for i in trues:
    train.append(truepath + i)
    labels[truepath + i] = 1
falsepath = 'dataset/6000samples/fold3/Train/False/'
falses = os.listdir(falsepath)
for i in falses:
    train.append(falsepath + i)
    labels[falsepath + i] = 0

partition['train'] = train


truepath = 'dataset/BalancedTest/True/'
trues = os.listdir(truepath)
for i in trues:
    test.append(truepath + i)
    labels[truepath + i] = 1
falsepath = 'dataset/BalancedTest/False/'
falses = os.listdir(falsepath)
for i in falses:
    test.append(falsepath + i)
    labels[falsepath + i] = 0

partition['test'] = test


class DataGenerator(keras.utils.Sequence):
	def __init__(self, list_IDs, labels, mean, std,  batch_size = 32, dim = (300,300), n_channels = 3, n_classes = 2, shuffle = True):
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()
		self.mean = mean
		self.std = std

	def  on_epoch_end(self):
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		X = np.empty([len(list_IDs_temp), *self.dim, self.n_channels])
		y = np.empty(len(list_IDs_temp), dtype = int)
		
		for i, ID in enumerate(list_IDs_temp):
			X[i] = img_to_array(load_img(ID, target_size = self.dim, interpolation = 'bicubic'))
			y[i] = self.labels[ID]
			
		X_e = np.zeros((X.shape[0],self.dim[0],self.dim[1], 1))
		
		for i in range(X.shape[0]):
			X_e[i,:,:,0] = sf.canny(0.2989*X[i,:,:,0] + 0.5870*X[i,:,:,1] + 0.1140*X[i,:,:,2])
		
				
		X = X[:,100:200,100:200,:]
		X_e = X_e[:,100:200,100:200,:]
		X = np.concatenate([X, X_e], axis = 3)
		X = (X - self.mean)/self.std
		
		y = np.reshape(y,(-1,1))
		dummy = np.zeros((y.shape[0],1))
		y = np_utils.to_categorical(y, self.n_classes)

		return X,y,dummy

	def __len__(self):
		return int(np.ceil(len(self.list_IDs) / self.batch_size)) 


	def __getitem__(self, index):
		indexes = self.indexes[index*self.batch_size:(index*self.batch_size) + min(   self.batch_size,   (len(self.list_IDs) - index * self.batch_size) )]
		list_IDs_temp = [self.list_IDs[k] for k in indexes]
		X, y, dummy = self.__data_generation(list_IDs_temp)
		return [X, y],[y, dummy]



params = {'list_IDs':partition['train'], 'labels':labels,  'dim': (300,300),'batch_size': trainsetsize,'n_classes': 2,'n_channels': 3,'shuffle': True, 'mean':0, 'std':1}
training_generator = DataGenerator(**params)
trainmean = 0.
trainstd = 0.
for a,b in training_generator:
	trainmean = np.mean(a[0])
	trainstd = np.std(a[0])
	print(trainmean, trainstd)
	break



testparams = {'list_IDs':partition['test'], 'labels':labels,  'dim': (300,300),'batch_size': testbatchsize, 'n_classes': 2,'n_channels': 3,'shuffle': True, 'mean':trainmean, 'std':trainstd}
test_generator = DataGenerator(**testparams)
##################



 
def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)

model=DenseNet()
model.compile(optimizer=Adam(lr=3e-4), loss=['categorical_crossentropy',zero_loss], loss_weights=[1, 0.1], metrics=['accuracy'])

model.load_weights('saswat_fold3.h5')
model_final=Model(inputs=model.get_layer("inputlayer").input,outputs=model.get_layer("outputlayer").output)


############################################# Testing ####################################################################################

model_final.compile(optimizer=Adam(lr=3e-4), loss='categorical_crossentropy', metrics=['accuracy'])
#looping=y_train.shape
#print(looping)
#mean=52.662945
#std=52.384068
'''
for i in range(looping[0]):
    y_pred=model_final.predict(x_train[i:i+1,:,:,:])
    print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)
    print(y_pred.shape)
    compa=np.argmax(y_train[i:i+1,:],axis=1)
    if y_pred==compa:
        print('correct prediction')
    else:
        if y_pred==0:
            print('incorrect prediction of non defect')
            img=x_train[i,:,:,:]*std
            img=img+mean
            img=img.astype('uint8')
            img=Image.fromarray(img,'RGB')
            path='./non_defected/'+str(i)+'non_defected.png'
            img.save(path)
        if y_pred==1:
            print('incorrect prediction of defect')
            img=x_train[i,:,:,:]*std
            img=img+mean
            img=img.astype('uint8')
            img=Image.fromarray(img,'RGB')
            path='./defected/'+str(i)+'defected.png'
            img.save(path)
'''
test_pred = 0.
test_gt = 0.
for a,b in test_generator:
	test_pred = model_final.predict(a[0])
	test_pred = np.argmax(test_pred, axis=1).astype(np.int32)
	test_gt = np.argmax(a[1],axis=1).astype(np.int32)
	break

tp,fp,tn,fn = getMetrics(test_gt,test_pred, 0 ,1)
testacc = (tp+tn)/(tp+fp+tn+fn)

ndp = tp/(tp+fp) if (tp+fp)>0. else 'Undefined'
ndr = tp/(tp+fn) if (tp+fn)>0. else 'Undefined'

tp,fp,tn,fn = getMetrics(test_gt,test_pred, 1 ,0)
dp = tp/(tp+fp) if (tp+fp)>0. else 'Undefined'
dr = tp/(tp+fn) if (tp+fn)>0. else 'Undefined'

print('Accuracy: {}'.format(testacc))
print('Defective Precision: {}'.format(dp))
print('Defective Recall: {}'.format(dr))
print('Non Defective Precision: {}'.format(ndp))
print('Non Defective Recall: {}'.format(ndr))



"""
target_names = ['Deffect 0', 'Non deffect 1']
print(classification_report(np.argmax(y_train,axis=1), y_pred,target_names=target_names))
print('Predicted .............................................................................')
"""


"""
######################confusion matrix definition ########################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    acc=np.trace(cm)/float(np.sum(cm))
    miss_class=1-acc
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    print(acc)
    print(miss_class)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i+0.25 if i==0 else i-0.25, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
###############################################################################################
cnf_matrix = (confusion_matrix(np.argmax(y_train,axis=1), y_pred))
np.set_printoptions(precision=2)
plt.figure(figsize=(10,10))
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
plt.savefig('confusion_matrix_test-2.png')

print('confusion matrix saved..............................................................')

from sklearn.metrics import*

print('Evaluate')
a=model_final.evaluate(x=x_train,y=y_train,batch_size=8,verbose=1)
print('Test_loss , Test_acc: ',a)
print('Predict')
out=model_final.predict(x=x_train, verbose=1, batch_size=8)


### obtain tpr, fpr and calculate AUROC
pred=out[:,0]
fpr, tpr, thresholds = roc_curve(np.argmax(y_train,axis=1), pred, pos_label=0)
# print('TPR: ',tpr)
# print('FPR: ',fpr)

precision = precision_score(np.argmax(y_train,axis=1),np.argmax(out,axis=1),pos_label=0, average='binary')
print('Precision: ',precision)

recall = recall_score(np.argmax(y_train,axis=1),np.argmax(out,axis=1),pos_label=0, average='binary')
print('Recall: ',recall)

f1score= (2*precision*recall)/(precision+recall)
print('F1score: ',f1score)

print('AUROC: ',auc(fpr, tpr))

## plot ROC
plt.plot(fpr,tpr)

"""
