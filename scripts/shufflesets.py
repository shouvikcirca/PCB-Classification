import numpy as np
import os

def permute(filelist):
    size = np.load(filelist[0]).shape[0]
    inds = np.random.permutation(size)

    for f in filelist:
        temp = np.load(f)
        temp = temp[inds]
        os.system('rm '+str(f))
        print('removed existing '+f)
        np.save(str(f), temp)
        print('saved new '+f)


if __name__ == '__main__':
    permute(['Xtrain.npy','Xtrain_e.npy','ytrain.npy'])
