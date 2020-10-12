import torch
import numpy
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os
import numpy as np

def tonp(rdir, target):
    finaltensor = torch.Tensor()
    count = 0
    images = os.listdir(rdir)
    for im in images:
        currim = load_img(rdir+'/'+im)
        currim = img_to_array(currim)
        imtensor = torch.from_numpy(currim)
        count+=1
        finaltensor = torch.cat([finaltensor, imtensor.unsqueeze(0)])   
        print(finaltensor.shape) 


    finalarray = finaltensor.numpy()
    truelabels = np.array([target for i in range(finalarray.shape[0])])
    with open(rdir+'.npy','wb') as f:
        np.save(f, finalarray)
    with open(rdir+'labels.npy','wb') as f:
        np.save(f, truelabels)


if __name__ == "__main__":
    tonp('False', 0)
    tonp('True',1)
