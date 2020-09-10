print('Enter Filename')
fname = input()



f = open(fname,'r')
paths = f.read().split('\n')
f.close()

csvfilename = fname[:-4]+'.csv'

f = open(csvfilename,'a')
for i in range(5):
    nstr=''
    print(paths[i])
    nstr+=(paths[i]+',')
    nstr = nstr + input()  # D or ND
    nstr+=','
    nstr = nstr + input()  # C or NC
    nstr+='\n'
    f.write(nstr)

f.close()

print('Data written to {}'.format(csvfilename))
