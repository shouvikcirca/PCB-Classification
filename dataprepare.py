import os
import numpy as np
from xmlparser import entries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img

class prepare:

    def getentries(self, xmlfile):
        with open(xmlfile,'r') as f:
        s = f.read()
        s = s.split('\n')
        s = s[4:-2]
        s = [i[5:-3] for i in s]
        s = [i.split(" ")[1:] for i in s]

        for i in range(len(s)):
            s[i][0] = s[i][0].split("=")[1]
            s[i][0] = s[i][0][1:-1]
            s[i][1] = s[i][1].split("=")[1]
            s[i][1] = s[i][1][1:-1]
            s[i][2] = s[i][2].split("=")[1]
            s[i][2] = s[i][2][1:-1]
        return s

    def formFolders(ents):
        # True:1 False:0
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







