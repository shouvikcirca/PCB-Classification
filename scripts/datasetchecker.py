from xmlparser import entries 
import os

ents = entries()
trues = os.listdir('True')
falses = os.listdir('False')

# Checking True and False folders with xml file
for i in range(len(entries())):
    if ents[i][2] == 'True':
        if ents[i][0]+'.tif' not in trues:
            print('invalid path 1')
    elif ents[i][2] == 'False':
        if ents[i][0]+'.tif' not in falses:
            print('invalid path 2')


# Checking True and False folders with targets.csv
f = open('targets.csv','r')
s = f.read().split('\n')[:-2]
f.close()
a = [i.split(',') for i in s]

for i in range(len(a)):
    if a[i][1] == 'True':
        if a[i][0]+'.tif' not in trues:
            print('invalid path 3')
    elif a[i][1] == 'False':
        if a[i][0]+'.tif' not in falses:
            print('invalid path 4')



