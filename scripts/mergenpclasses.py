import numpy as np

def mergeclasses(filelist, outputfilename):
    sizelist = [np.load(i).shape[0] for i in filelist]
    print(sizelist)
    
    totalsize = 0
    for i in sizelist:
        totalsize+=i
    print(totalsize)

    finalarraysize = list(np.load(filelist[0]).shape)
    finalarraysize[0] = totalsize
    finalarray = np.zeros(finalarraysize)
    print(finalarray.shape)

    fcounter = 0
    for f in filelist:
        temp = np.load(f)
        for im in temp:
            print(fcounter)
            finalarray[fcounter] = im
            fcounter+=1

    np.save(outputfilename,finalarray)





if __name__ == '__main__':
    #mergeclasses(['False.npy','True.npy'], 'X.npy')
    #mergeclasses(['False_e.npy','True_e.npy'], 'X_e.npy')
    mergeclasses(['Falselabels.npy','Truelabels.npy'], 'y.npy')



        

    
