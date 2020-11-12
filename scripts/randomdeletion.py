import numpy as np
import os

falseims = os.listdir('False')
trueims = os.listdir('True')

flen = len(falseims)
tlen = len(trueims)
diff = flen - tlen

a = np.random.permutation(flen)
for i in range(diff):
    execline = 'False/'+ str(falseims[i])
    print(execline)
    os.system('rm '+execline)


print(len(os.listdir('False')))
print(len(os.listdir('True')))
