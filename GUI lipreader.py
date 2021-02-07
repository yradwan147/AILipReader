import tkinter as tk
from PIL import Image, ImageTk
import os

def run_preprocessing_script():
    os.system('finalpre.py')
def open_camera():
    os.system('IPCAM2f.py')
def run_test_script():
    os.system('finaltest.py')

master = tk.Tk()

background_image=tk.PhotoImage(file = "Button.png")
    
load = Image.open("stemlogo.png")
load = load.resize((100,115))
render = ImageTk.PhotoImage(load)
img_left = tk.Label(master, image=render)
img_left.image = render
img_left.grid(column = 0, row = 0, padx=(20, 0))

load2 = Image.open("icon.png")
load2 = load2.resize((150,150))
render2 = ImageTk.PhotoImage(load2)
img_right = tk.Label(master, image=render2)
img_right.image = render2
img_right.grid(column = 4, row = 0, padx=(0, 20))

basic_text = "Artificial lip-reader" + "\nV1.0.0"

basic_data1 = tk.Label(master, text = basic_text)
basic_data1.config(font=("Courier", 24))
basic_data1.grid(column = 2, row=0)
basic_data2 = tk.Label(master, text = " ")
basic_data2.grid(column = 2, row=2)


var2 = tk.IntVar()

next_button = tk.Button(master, text='Preprocess \nData', command=run_preprocessing_script, width = 15)
next_button.grid(column = 1, row=3, pady=(0, 20))
next_button2 = tk.Button(master, text='Open \nCamera', command=open_camera, width = 15)
next_button2.grid(column = 2, row=3, pady=(0, 20))
next_button3 = tk.Button(master, text='Test', command=run_test_script, width = 15)
next_button3.grid(column = 3, row=3, pady=(0, 20))


