import math
import os
from keras.preprocessing.image import load_img
from tqdm import tqdm

def createSegs(dirname, folds):
    dirfiles = os.listdir(dirname)
    testfiles = dirfiles[ math.floor(0.8*len(dirfiles)):]
    trainvalfiles = dirfiles[:math.floor(0.8*len(dirfiles))]

    total = len(trainvalfiles)
    flimits = [0]
    for i in range(folds):
        flimits.append( min(int((i+1)*(total/folds)), total) )
    pwdfiles = os.listdir()

    
    os.mkdir('Test/'+dirname)

    for k in tqdm(testfiles):
        im = load_img(dirname+'/'+k)
        im.save('Test/'+dirname+'/'+k)
        #print('Test/'+dirname+'/'+k)


    for i in tqdm(range(folds)):
        if 'Seg'+str(i+1) not in pwdfiles:
            os.mkdir('Seg'+str(i+1))
        os.mkdir('Seg'+str(i+1)+'/'+ dirname)
        for j in trainvalfiles[flimits[i]:flimits[i+1]]:
            im = load_img(dirname+'/'+j)
            im.save('Seg'+str(i+1)+'/'+ dirname+'/'+j)
            #print('Seg'+str(i+1)+'/'+dirname+'/'+j)




def createFolds(folds):

    flist = [i+1 for i in range(folds)]

    for i in flist:
        os.mkdir('Fold'+str(i))
        os.mkdir('Fold'+str(i)+'/'+'Train')
        os.mkdir('Fold'+str(i)+'/'+'Validation')
        os.mkdir('Fold'+str(i)+'/'+'Train/True')
        os.mkdir('Fold'+str(i)+'/'+'Train/False')
        os.mkdir('Fold'+str(i)+'/'+'Validation/True')
        os.mkdir('Fold'+str(i)+'/'+'Validation/False')


    for i in flist:
        nlist = [j for j in flist if j!=i]
        
        for k in tqdm(nlist):
            os.system('cp Seg'+str(k)+'/True/* Fold'+str(i)+'/Train/True')
            os.system('cp Seg'+str(k)+'/False/* Fold'+str(i)+'/Train/False')

        os.system('cp Seg'+str(i)+'/False/* Fold'+str(i)+'/Validation/False')
        os.system('cp Seg'+str(i)+'/True/* Fold'+str(i)+'/Validation/True')

    for i in range(folds):    
        os.system('rm -rf Seg'+str(i+1))
    


if __name__ == '__main__':
    os.mkdir('Test')
    createSegs('False',3)
    createSegs('True',3)
    createFolds(3)




