from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import os
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
import cv2 

trues = os.listdir('True')
falsecount = len(os.listdir('False'))

diff = falsecount - len(trues)
print(diff)


for i in range(diff):
    a = load_img('True/'+trues[i])
    b = cv2.flip(img_to_array(a), 0)
    b = array_to_img(b)
    print(trues[i])
    b.save('True/'+trues[i][:-4]+'_aug.tif')
