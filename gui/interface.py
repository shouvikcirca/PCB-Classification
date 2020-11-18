from tkinter import *
from tkinter.filedialog import *
from PIL import ImageTk, Image


root = Tk()
root.title('Image Classifier')

def load1():
    f1 = askopenfilename(title = 'open').split('/')[-1]
    print(f1)
    im = Image.open(f1)
    im = im.resize((250, 250), Image.ANTIALIAS)
    im = ImageTk.PhotoImage(im)
    Label(root, image=im).grid(row = 1, column = 0, columnspan = 2)
    c = StringVar()
    c.set('Defective')
    Label(root, textvariable = c).grid(row = 0, column = 1, sticky = W)
    panel.image = im



Button(root, text = 'Choose Image', command = load1).grid(row = 0, column = 0)

for child in root.winfo_children():
    child.grid_configure(padx = 10, pady = 10)

root.mainloop()
