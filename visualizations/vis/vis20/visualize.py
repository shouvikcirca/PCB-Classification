import matplotlib.pyplot as plt
import os
import numpy as np


exp1 = 'logs'+'14'
exp2 = 'logs'+'46'


fold1_1 = np.load(exp1+'/logs'+'/fold1/vallossrecord.npy')
fold2_1 = np.load(exp1+'/logs'+'/fold2/vallossrecord.npy')
fold3_1 = np.load(exp1+'/logs'+'/fold3/vallossrecord.npy')


fold1_2 = np.load(exp2+'/logs'+'/fold1/vallossrecord.npy')
fold2_2 = np.load(exp2+'/logs'+'/fold2/vallossrecord.npy')
fold3_2 = np.load(exp2+'/logs'+'/fold3/vallossrecord.npy')


fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(1,3)


a = fig.add_subplot(gs[0,0])
b = fig.add_subplot(gs[0,1])
c = fig.add_subplot(gs[0,2])


a.plot(fold1_1, label='resnext50_32x4d')
a.plot(fold1_2, label='self attention with elements')



b.plot(fold2_1, label='resnext50_32x4d')
b.plot(fold2_2, label='self attention with elements')


c.plot(fold3_1, label='resnext50_32x4d')
c.plot(fold3_2, label='self attention with elements')

a.legend(loc='upper right')
b.legend(loc='upper right')
c.legend(loc='upper right')

a.title.set_text('Fold 1')
a.set_xlabel('epochs')
a.set_ylabel('Validation loss')


b.title.set_text('Fold 2')
b.set_xlabel('epochs')
b.set_ylabel('Validation loss')


c.title.set_text('Fold 3')
c.set_xlabel('epochs')
c.set_ylabel('Validation loss')

fig.tight_layout(pad=3.0)

plt.show()








