import tensorflow as tf
if tf.__version__[0] == '2':
    from tensorflow.keras import models
    from tensorflow.keras import layers
    from tensorflow.keras.applications import DenseNet121

else:
    from keras import models
    from keras import layers
    from keras.applications import DenseNet121

    
def dn121scratch():
    conv_base = DenseNet121(include_top = False, pooling='avg', input_shape=(600,600,3))
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Dense(1, activation = 'sigmoid'))

    return model


