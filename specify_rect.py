#!/usr/bin/python3
import tkinter as tk
from tkinter import messagebox as mb
from PIL import Image, ImageTk

class SelectRegionApp(tk.Tk):
    def __init__(self, image_filename, result):
        tk.Tk.__init__(self)
        self.result_dict = result
        self.x = self.y = 0
        self.im = Image.open(image_filename)
        self.tk_im = ImageTk.PhotoImage(self.im)

        self.label = tk.Label(self, text="Select a Rectangle To Extract")
        self.label.pack(side="top")


        self.canvas = tk.Canvas(self, width=self.tk_im.width(), height=self.tk_im.height(), cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.rect = None
        self.start_x = None
        self.start_y = None

        self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)

        self.button = tk.Button(self, text="DONE", command=self.done)
        self.button.pack(side="bottom")

    def done(self):
        if self.start_x is None:
            mb.showwarning("warning","you need to drag a rectangle over the region you want to extract before continuing")
        self.result_dict["rect"] = self.canvas.coords(self.rect)
        self.destroy()

    def on_button_press(self, event):
        if not self.rect is None:
            self.canvas.delete(self.rect)
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y

        # create rectangle if not yet exist
        #if not self.rect:
        self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, fill="")

    def on_move_press(self, event):
        curX, curY = (event.x, event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        pass


def select_rectangle(image_filename):
    ans = dict()
    app = SelectRegionApp('/home/cohend/Dropbox/Photos/David3.jpeg', ans)
    app.mainloop()
    return ans

print(select_rectangle('/home/cohend/Dropbox/Photos/David3.jpeg'))




# if __name__ == "__main__":
#     app = SelectRegionApp('/home/cohend/Dropbox/Photos/David3.jpeg')
#     app.mainloop()
