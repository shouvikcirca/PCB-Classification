import numpy as np
import skimage.feature as sf

def cannify(filename):   
    a = np.load(filename)
    c = []

    for i in a:
        c.append(sf.canny(0.2989*i[:,:,0]+0.5870*i[:,:,1]+0.1140*i[:,:,2]))
        print(len(c))
    print(np.shape(c))
    c=np.asarray(c)
    np.save(filename[:-4]+"_e.npy",c)
    print('Written '+filename[:-4]+"_e.npy")


if __name__ == '__main__':
    cannify('True.npy')
    cannify('False.npy')
