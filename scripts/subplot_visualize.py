import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

images = os.listdir()
images = [i for i in images if 'a.py' not in i]

ims = []
for i in range(6):
    ims.append(plt.imread(images[i]))

ims = ims[:6]

ims100 = [i for i in ims]
for i in range(6):
    ims100[i] = np.asarray(ims100[i])
    ims100[i] = ims100[i][250:350,250:350,:]

ims200 = [i for i in ims]
for i in range(6):
    ims200[i] = np.asarray(ims200[i])
    ims200[i] = ims200[i][200:400,200:400,:]



fig = plt.figure(figsize = (10,10))
gs = fig.add_gridspec(6, 3)


implots = []
for i in range(6):
    implot = []
    for j in range(3):
        implot.append(fig.add_subplot(gs[i,j]))
    implots.append(implot)


for i in range(6):
        implots[i][0].imshow(ims[i])
        implots[i][1].imshow(ims100[i])
        implots[i][2].imshow(ims200[i])

plt.show()

