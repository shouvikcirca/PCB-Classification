import os

print('Enter Filename')
fname = input()


f = open(fname,'r')
paths = f.read().split('\n')
f.close()

csvfilename = fname[:-4]+'.csv'

nstr = ''
for i in range(3):
    print(paths[i])
    os.system('eog ../allimages/'+paths[i])
    nstr+=(paths[i]+',')
    nstr = nstr + input()  # D or ND
    nstr+=','
    nstr = nstr + input()  # C or NC
    nstr+='\n'

f = open(csvfilename,'a')
f.write(nstr)
f.close()

print('Data written to {}'.format(csvfilename))
