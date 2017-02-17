import tkinter as tk
from tkinter import messagebox as mb
from PIL import Image, ImageTk

class LabelImageBBOXApp(tk.Tk):
    def __init__(self, image_filename, bbox, image_resize, result):
        tk.Tk.__init__(self)
        self.result_dict = result
        self.x = self.y = 0
        im = Image.open(image_filename)
        if not image_resize is None:
            im = im.resize(image_resize)
        cropped = im.crop(bbox)
        self.tk_im = ImageTk.PhotoImage(cropped)

        self.label = tk.Label(self, text="Label this image region")
        self.label.pack(side="top")

        self.l2 = tk.Label(self, image=self.tk_im)
        self.l2.pack()

        self.e = tk.Entry(self)
        self.e.pack()

        self.button = tk.Button(self, text="Submit Label", command=self.done)
        self.button.pack(side="bottom")

        self.wm_title(image_filename)
        self.e.bind('<Return>', self.done)
        self.e.focus_set()
        
    def done(self, event=None):
        if self.e.get() == '':
            mb.showwarning("warning","we don't accept blank labels")
        else:
            self.result_dict["label"] = self.e.get()
            self.destroy()


def manually_label(image_filename, bbox, image_size = None):
    ans = dict()
    app = LabelImageBBOXApp(image_filename, bbox, image_size, ans)
    app.mainloop()
    return ans['label']

if __name__=="__main__":
    print(manually_label("/home/cohend/Dropbox/Photos/David.jpeg", (1,1,50,50)))
