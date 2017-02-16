import tkinter as tk
from tkinter import messagebox as mb
from PIL import Image, ImageTk

class LabelImageBBOXApp(tk.Tk):
    def __init__(self, image_filename, bbox, result):
        tk.Tk.__init__(self)
        self.result_dict = result
        self.x = self.y = 0
        im = Image.open(image_filename)
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

    def done(self):
        if self.e.get() == '':
            mb.showwarning("warning","we don't accept blank labels")
        else:
            self.result_dict["label"] = self.e.get()
            self.destroy()


def manually_label(image_filename, bbox):
    ans = dict()
    app = LabelImageBBOXApp(image_filename, bbox, ans)
    app.mainloop()
    return ans['label']

if __name__=="__main__":
    print(manually_label("/home/cohend/Dropbox/Photos/David.jpeg", (1,1,50,50)))
