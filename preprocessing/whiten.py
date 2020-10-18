from keras.preprocessing.image import NumpyArrayIterator
import numpy as np

def whiten(X,y,conv_base, datagen = None, pooling=False, shuffle=True,seed=123):
    nos = X.shape[0]
    bs = int(nos/8)

    numit = NumpyArrayIterator(
        X, y,image_data_generator=datagen,  batch_size=bs, shuffle=shuffle, sample_weight=None,
        seed=seed, data_format=None, save_to_dir=None, save_prefix='',
        save_format='png', subset=None, dtype=None
    )

    exf = conv_base.predict(X[0].reshape(1,X.shape[1],X.shape[2],X.shape[3])).shape
    
    if pooling is not True:
        samples = np.zeros(shape=(nos,exf[1],exf[2],exf[3]))
    else:
        samples = np.zeros(shape=(nos,exf[1]))
    labels = np.zeros(shape=(nos))
    i = 0
    for samples_batch,labels_batch in numit:
        f = conv_base.predict(samples_batch)
        print(f.shape)
        samples[i*bs:(i+1)*bs] = f
        labels[i*bs:(i+1)*bs] = labels_batch
        i+=1
        if (i*bs)>=nos:
            break

    return samples, labels
