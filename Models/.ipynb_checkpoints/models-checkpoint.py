import tensorflow as tf

if tf.__version__[0] == '2':
    from tensorflow.keras import models
    from tensorflow.keras import layers
    from tensorflow.keras.applications import DenseNet121
    import efficientnet.tfkeras as efn

else:
    from keras import models
    from keras import layers
    from keras.applications import DenseNet121
    import efficientnet
    
def densenet(l):
    if l == 121:
        conv_base = DenseNet121(include_top = False, pooling='avg', input_shape=(600,600,3))
    elif l == 169:
        conv_base = DenseNet169(include_top = False, pooling='avg', input_shape=(600,600,3))
    else:
        conv_base = DenseNet201(include_top = False, pooling='avg', input_shape=(600,600,3))
    
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Dense(1, activation = 'sigmoid'))
    return model


def efficientnet(v):
    if v == 0:
        conv_base = efn.EfficientNetB0(weights='imagenet',include_top=False, pooling='avg', input_shape=(600,600,3))
    elif v == 1:
        conv_base = efn.EfficientNetB1(weights='imagenet',include_top=False, pooling='avg', input_shape=(600,600,3))
    elif v == 2:
        conv_base = efn.EfficientNetB2(weights='imagenet',include_top=False, pooling='avg', input_shape=(600,600,3))
    elif v == 3:
        conv_base = efn.EfficientNetB3(weights='imagenet',include_top=False, pooling='avg', input_shape=(600,600,3))
    elif v == 4:
        conv_base = efn.EfficientNetB4(weights='imagenet',include_top=False, pooling='avg', input_shape=(600,600,3))
    elif v == 5:
        conv_base = efn.EfficientNetB5(weights='imagenet',include_top=False, pooling='avg', input_shape=(600,600,3))
    elif v == 6:
        conv_base = efn.EfficientNetB6(weights='imagenet',include_top=False, pooling='avg', input_shape=(600,600,3))
    else:
        conv_base = efn.EfficientNetB7(weights='imagenet',include_top=False, pooling='avg', input_shape=(600,600,3))

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Dense(1, activation = 'sigmoid'))
    return model
