import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import array_to_img

for i in range(5):
    a = np.random.uniform(0,255,[32,32,3])
    a = array_to_img(a)
    print(str(int(np.random.uniform(0,255,1)))+'.tif')
    a.save(str(int(np.random.uniform(0,255,1)))+'.tif')

