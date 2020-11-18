from tkinter import *
from tkinter.filedialog import *
from PIL import ImageTk, Image
from Models.models import densenet
from keras.preprocessing.image import load_img, img_to_array
import numpy as np


print('Loading model')
model = densenet(201)

root = Tk()
root.title('Image Classifier')

def load1():
    f1 = askopenfilename(title = 'open').split('/')[-1]
    print(f1)
    loadedim = np.expand_dims(img_to_array(load_img(f1)),axis = 0)
    pred = (model.predict(loadedim)>=0.5).astype('int32').reshape(-1)[0]
    print(pred)
    c = StringVar()
    if pred == 0:
        c.set('Non Defective')
    else:
        c.set('Defective')

    im = Image.open(f1)
    im = im.resize((250, 250), Image.ANTIALIAS)
    im = ImageTk.PhotoImage(im)
    panel = Label(root, image = im).grid(row = 1, column = 0, columnspan = 2)
    Label(root, textvariable = c).grid(row = 0, column = 1, sticky = W)
    panel.image = im



Button(root, text = 'Choose Image', command = load1).grid(row = 0, column = 0)

for child in root.winfo_children():
    child.grid_configure(padx = 10, pady = 10)

root.mainloop()
