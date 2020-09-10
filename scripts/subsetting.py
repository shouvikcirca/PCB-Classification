import os

files = os.listdir('allimages')


fileprefix = 'subset'
subsetindex = 1

for i in range(len(files)):
    ns = files[i]+'\n'
    f = open(fileprefix+str(subsetindex)+'.txt','a')
    f.write(ns)
    f.close()
    if i>0 and i%100 == 0:
        subsetindex+=1



