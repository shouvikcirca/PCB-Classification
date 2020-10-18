from keras import models
from keras import layers
from keras.applications import DenseNet121

def dn121scratch(X):
    conv_base = DenseNet121(include_top = False, pooling='avg', input_shape=(X.shape[1],X.shape[2],X.shape[3]))
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Dense(1, activation = 'sigmoid'))

    return model


