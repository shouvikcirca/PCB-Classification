import numpy as np
import matplotlib.pyplot as plt

trainloss = np.load('trainlossrecord.npy')
valloss = np.load('vallossrecord.npy')

print(trainloss.shape)
print(valloss.shape)


plt.plot(trainloss, color = 'r', label = 'Train Loss')
plt.plot(valloss, color = 'b', label = 'Val Loss')
plt.legend()
plt.show()
