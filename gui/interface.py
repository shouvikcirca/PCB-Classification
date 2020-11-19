from tkinter import *
from tkinter.filedialog import *
from tkinter import ttk
from PIL import ImageTk, Image
#from Models.models import densenet
#from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

#print('Loading model')
#model = densenet(201)



root = Tk()
root.title('Image Classifier')
root.geometry("600x600") 

dpreds = ''
ndpreds = ''
lb1 = ''
lb2 =''
scroll1 = ''
scroll2 = ''
dlabel = ''
ndlabel = ''
panel = ''
lb = ''

def load1():
    global panel
    global lb
    f1 = askopenfilename(title = 'open')
    print(f1)
    #loadedim = np.expand_dims(img_to_array(load_img(f1))/255.,axis = 0)
    #pred = (model.predict(loadedim)>=0.5).astype('int32').reshape(-1)[0]
    #print(pred)
    pred = np.random.randint(2)
    c = StringVar()
    if pred == 0:
        c.set('Non Defective')
    else:
        c.set('Defective')

    if 'str' not in str(type(panel)):
        panel.destroy()
    if 'str' not in str(type(lb)):
        lb.destroy()

    im = Image.open(f1)
    im = im.resize((400, 400), Image.ANTIALIAS)
    im = ImageTk.PhotoImage(im)
    panel = Label(myframe1, image = im)#.grid(row = 1, column = 0, columnspan = 2)
    lb = Label(myframe1, textvariable = c)#.grid(row = 0, column = 1, sticky = W)
    panel.image = im
    lb.pack(side = 'top')
    panel.pack(side = 'top')


def loadfolder():
    global dpreds
    global ndpreds

    f1 = askopenfilename(title = 'open').split('/')[:-1]
    f1 = '/'.join(f1)
    ims = os.listdir(f1)
    dlist = [];ndlist = [];
    for i in range(len(ims)):
        if np.random.randint(2) == 0:
            dlist.append(ims[i])
        else:
            ndlist.append(ims[i])

    dpreds = dlist
    ndpreds = ndlist
    #dpreds = 'Defective\n\n'+'\n'.join(dlist)
    #ndpreds = 'NonDefective\n\n'+'\n'.join(ndlist)


def showPredictions():
    global dpreds
    global ndpreds
    global myframe3
    global lb1
    global lb2
    global dlabel
    global ndlabel
    global scroll1
    global scroll2

    d = StringVar()
    d.set(dpreds)
    nd = StringVar()
    nd.set(ndpreds)
   
    dlabel = Label(myframe3, text='Defective')#.grid(row = 1, column = 0, columnspan = 2)
    dlabel.pack(side = 'left')

    ndlabel = Label(myframe3, text = 'NonDefective')#.grid(row = 1, column = 0, columnspan = 2)
    ndlabel.pack(side = 'right')

    scroll1 = Scrollbar(myframe3)
    scroll1.pack(side = 'left')
    lb1 = Listbox(myframe3, yscrollcommand = scroll1.set)

    scroll2 = Scrollbar(myframe3)
    scroll2.pack(side = 'right')
    lb2 = Listbox(myframe3, yscrollcommand = scroll2.set)
    lb1.pack(side = LEFT, expand = 1, fill = 'both')
    lb2.pack(side = RIGHT, expand = 1, fill = 'both')

    for i in dpreds:
        lb1.insert(END, str(i))
    for i in ndpreds:
        lb2.insert(END, str(i))
    
    scroll1.config(command = lb1.yview)
    scroll2.config(command = lb2.yview)



def delPreds():
    global dpreds
    global ndpreds
    global myframe3
    global lb1
    global lb2
    global dlabel
    global ndlabel
    global scroll1
    global scroll2
    
    lb1.destroy()
    lb2.destroy()
    dlabel.destroy()
    ndlabel.destroy()
    scroll1.destroy()
    scroll2.destroy()





mynotebook = ttk.Notebook(root)
myframe1 = Frame(mynotebook)
myframe2 = Frame(mynotebook)
myframe3 = Frame(mynotebook)


mynotebook.add(myframe1,text='Solo')
mynotebook.add(myframe2,text='Group')
mynotebook.add(myframe3,text='Show Results')

mynotebook.pack(expand = 1, fill = 'both')


b1 = Button(myframe1, text = 'Choose Image', command = load1)#.grid(row = 0, column = 0)
b2 = Button(myframe2, text = 'Select Folder', command = loadfolder)#.grid(row = 0, column = 0)
b3 = Button(myframe3, text = 'Show Predictions', command = showPredictions)#.grid(row = 0, column = 0)
b4 = Button(myframe3, text = 'Delete Predictions', command = delPreds)

b1.pack(side = 'top')
b2.pack(side = 'top')
b3.pack(side = 'top')
b4.pack(side = 'top')


root.mainloop()
