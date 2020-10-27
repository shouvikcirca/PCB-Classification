import matplotlib.pyplot as plt

def display(bwimage):
    try:
        plt.imshow(bwimage, cmap=plt.cm.gray)
        plt.show()
    except:
        print("Error:input has to be a numoy array")


if __name__ == '__main__':
    display(1)
