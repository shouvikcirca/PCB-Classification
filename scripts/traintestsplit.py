import numpy as np


def splitup(npyfile, labelfile, testfraction):
    points = np.load(npyfile)
    trainfraction = 1 - testfraction
    train = points[:int(trainfraction*points.shape[0])]
    print("formed train")
    test = points[int(trainfraction*points.shape[0]):]
    print("formed test")
    label = np.load(labelfile)
    label = label[0]
    trainlabels = np.array([label for i in range(train.shape[0])])
    testlabels = np.array([label for i in range(test.shape[0])])
    with open("train"+npyfile,"wb") as f:
        np.save(f,train)
    print("written train array to file")
    with open("test"+npyfile,"wb") as f:
        np.save(f,test)
    print("written test array to file")
    with open("trainlabels"+npyfile,"wb") as f:
        np.save(f, trainlabels)
    print("written train labels")
    with open("testlabels"+npyfile,"wb") as f:
        np.save(f, testlabels)
    print("written test labels")



if __name__ == "__main__":
    splitup("False.npy", "Falselabels.npy", 0.2)
    splitup("True.npy", "Truelabels.npy", 0.2)



