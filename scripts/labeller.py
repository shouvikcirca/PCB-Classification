import os

print('Enter Filename')
fname = input()
allimages = os.listdir('../allimages/')


f = open(fname,'r')
paths = f.read().split('\n')
f.close()

csvfilename = fname[:-4]+'.csv'

f = open(csvfilename,'a')
for i in range(len(paths)):
    nstr=''
    print(paths[i])
    for ims in allimages:
        if paths[i] == ims:
            os.system('eog ../allimages/'+ims)
    nstr+=(paths[i]+',')
    nstr = nstr + input()  # D or ND
    nstr+=','
    nstr = nstr + input()  # C or NC
    nstr+='\n'
    f.write(nstr)

f.close()

print('Data written to {}'.format(csvfilename))
