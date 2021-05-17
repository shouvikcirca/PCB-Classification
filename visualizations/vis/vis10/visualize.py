import matplotlib.pyplot as plt
import os
import numpy as np


exp1 = 'logs'+'14'
exp2 = 'logs'+'56'
f = str(1)


fold1_1 = np.load(exp1+'/logs'+'/fold1/vallossrecord.npy')
fold2_1 = np.load(exp1+'/logs'+'/fold2/vallossrecord.npy')
fold3_1 = np.load(exp1+'/logs'+'/fold3/vallossrecord.npy')


fold1_2 = np.load(exp2+'/logs'+'/fold1/vallossrecord.npy')
fold2_2 = np.load(exp2+'/logs'+'/fold2/vallossrecord.npy')
fold3_2 = np.load(exp2+'/logs'+'/fold3/vallossrecord.npy')



fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(1,3)


a = fig.add_subplot(gs[0,0])
b = fig.add_subplot(gs[0,0])
c = fig.add_subplot(gs[0,1])
d = fig.add_subplot(gs[0,1])
e = fig.add_subplot(gs[0,2])
f = fig.add_subplot(gs[0,2])


a.plot(fold1_1, label='resnext50_32x4d with T')
b.plot(fold1_2, label = 'resnext50_32x4d without T')
c.plot(fold2_1, label='resnext50_32x4d with T')
d.plot(fold2_2, label = 'resnext50_32x4d without T')
e.plot(fold3_1, label='resnext50_32x4d with T')
f.plot(fold3_2, label = 'resnext50_32x4d without T')

a.legend()
c.legend()
e.legend(loc='upper left')

a.title.set_text('Fold 1')
c.title.set_text('Fold 2')
f.title.set_text('Fold 3')




a.set_xlabel('epochs')
c.set_xlabel('epochs')
f.set_xlabel('epochs')
a.set_ylabel('Validation loss')
c.set_ylabel('Validation loss')
f.set_ylabel('Validation loss')






fig.tight_layout(pad=3.0)

plt.show()








