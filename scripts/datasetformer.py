import os
import numpy as np
from xmlparser import entries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img


# True:1 False:0
ents = entries()
data = [i[1] for i in ents]
labels = np.array([1 if i[2]=='True' else 0 for i in ents])

s = ''
npdata = np.zeros((len(ents),600,600,3), dtype=float)
for i in range(len(ents)):
    print(ents[i][0],ents[i][2])
    a = data[i].split('\\')
    a = a[4:]
    a = '/'.join(a)
    img = load_img(a)
    imgarray = img_to_array(img)
    if(labels[i] == 1):
        save_img(path='True/'+ents[i][0]+'.tif', x=imgarray)
    else:
        save_img(path='False/'+ents[i][0]+'.tif', x=imgarray)

    s = s+ents[i][0]+','+ents[i][2]+'\n'

with open('targets.csv','w') as f:
    f.write(s)


