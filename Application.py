import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import filedialog, ttk, messagebox
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

# Load model
model = load_model('train.h5')

def browseFiles():
    global f
    f = filedialog.askopenfilename(
        initialdir="/",
        title="Select a CSV File",
        filetypes=(("CSV files", "*.csv*"), ("All files", "*.*"))
    )
    label_file_explorer.configure(text="File Opened: " + f)

def start():
    global f
    if 'f' not in globals():
        messagebox.showerror("Error", "No file selected!")
        return

    print("Process Started")
    dataset = pd.read_csv(f)
    print(dataset.info())

    X = dataset.iloc[:, :].values
    ypred = np.argmax(model.predict(X), axis=-1)  # Fix for deprecated `predict_classes`

    # Show result in new window
    result_window = Toplevel(window)
    result_window.title("Result for Sensor Data")
    ttk.Label(result_window, text=f"Prediction: {ypred[0]}", font=("Arial", 14)).pack(padx=20, pady=30)

# Main Window
window = Tk()
window.title('Application')
window.geometry("700x400")
window.config(background="white")

# Background Image
image1 = Image.open("bg.jpg")
test = ImageTk.PhotoImage(image1)
label1 = Label(window, image=test)
label1.image = test
label1.place(x=0, y=0)

# Labels and Buttons
label_file_explorer = Label(window, text="Please give Input Sensor data", width=100, height=4, fg="blue")
label_file_explorer.grid(column=1, row=1, padx=1, pady=5)

button_explore = Button(window, text="Browse Input Sensor Data", command=browseFiles, height=5)
button_explore.grid(column=1, row=2, padx=5, pady=5)

button_exit = Button(window, text="Exit", command=window.quit, height=5, width=10)
button_exit.grid(column=1, row=3, padx=5, pady=5)

button_start = Button(window, text="Start Analyzing Request", command=start, height=5)
button_start.grid(column=1, row=4, padx=5, pady=5)

window.mainloop()
