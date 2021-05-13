import matplotlib.pyplot as plt
import os
import numpy as np


exp1 = 'logs'+'14'
exp2 = 'logs'+'24'
exp3 = 'logs'+'25'
exp4 = 'logs'+'26'
exp5 = 'logs'+'27'


fold1_1 = np.load(exp1+'/logs'+'/fold1/vallossrecord.npy')
fold2_1 = np.load(exp1+'/logs'+'/fold2/vallossrecord.npy')
fold3_1 = np.load(exp1+'/logs'+'/fold3/vallossrecord.npy')


fold1_2 = np.load(exp2+'/logs'+'/fold1/vallossrecord.npy')
fold2_2 = np.load(exp2+'/logs'+'/fold2/vallossrecord.npy')
fold3_2 = np.load(exp2+'/logs'+'/fold3/vallossrecord.npy')


fold1_3 = np.load(exp3+'/logs'+'/fold1/vallossrecord.npy')
fold2_3 = np.load(exp3+'/logs'+'/fold2/vallossrecord.npy')
fold3_3 = np.load(exp3+'/logs'+'/fold3/vallossrecord.npy')



fold1_4 = np.log(np.load(exp4+'/logs'+'/fold1/vallossrecord.npy'))
fold2_4 = np.load(exp4+'/logs'+'/fold2/vallossrecord.npy')
fold3_4 = np.load(exp4+'/logs'+'/fold3/vallossrecord.npy')



fold1_5 = np.log(np.load(exp5+'/logs'+'/fold1/vallossrecord.npy'))
fold2_5 = np.load(exp5+'/logs'+'/fold2/vallossrecord.npy')
fold3_5 = np.load(exp5+'/logs'+'/fold3/vallossrecord.npy')











fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(1,3)


a = fig.add_subplot(gs[0,0])
b = fig.add_subplot(gs[0,1])
c = fig.add_subplot(gs[0,2])


a.plot(fold1_1, label='resnext50_32x4d')
a.plot(fold1_2, label='m = 2')
a.plot(fold1_3, label='m = 3')
a.plot(fold1_4, label='m = 4')
a.plot(fold1_5, label='m = 5')

a.set_xlabel('epochs')
a.set_ylabel('Validation loss')


b.plot(fold2_1, label='resnext50_32x4d')
b.plot(fold2_2, label='m = 2')
b.plot(fold2_3, label='m = 3')
b.plot(fold2_4, label='m = 4')
b.plot(fold2_5, label='m = 5')

b.set_xlabel('epochs')
b.set_ylabel('Validation loss')

c.plot(fold3_1, label='resnext50_32x4d')
c.plot(fold3_2, label='m = 2')
c.plot(fold3_3, label='m = 3')
c.plot(fold3_4, label='m = 4')
c.plot(fold3_5, label='m = 5')

c.set_xlabel('epochs')
c.set_ylabel('Validation loss')




a.legend(loc='upper right')
b.legend(loc='upper right')
c.legend(loc='upper right')

a.title.set_text('Fold 1')
b.title.set_text('Fold 2')
c.title.set_text('Fold 3')

fig.tight_layout(pad=3.0)

plt.show()








