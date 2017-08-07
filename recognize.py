import os, sys

import tensorflow as tf

from fileio import mnist_read, misc
from conv2d import model

import numpy as np

from PIL import Image, ImageDraw, ImageTk
import tkinter

SAVE_PATH = 'X:/mnist/model/conv2d/model-epoch'
SAVE_DIR = '/'.join(SAVE_PATH.split('/')[0:-1])
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)

in_x = tf.placeholder(dtype=tf.float32, shape=[mnist_read.IMAGE_WIDTH, mnist_read.IMAGE_HEIGHT])
out = model.create_conv2d_model(in_x)
saver = tf.train.Saver()
saved_epochs = misc.get_highest_epoch_saved(SAVE_DIR)
if saved_epochs == 0:
    sys.exit('Train the model first')
session = tf.Session()
saver.restore(sess=session, save_path=(SAVE_PATH + '-' + str(saved_epochs)))

# GUI Definitions
GUI_IMAGE_WIDTH, GUI_IMAGE_HEIGHT = 150, 150
image = Image.new("1", (GUI_IMAGE_WIDTH, GUI_IMAGE_HEIGHT))
paint = ImageDraw.Draw(image)
paint.rectangle([0, 0, GUI_IMAGE_WIDTH, GUI_IMAGE_HEIGHT], fill='white')

window = tkinter.Tk()

canvas = tkinter.Canvas(window, bg='white', width=GUI_IMAGE_WIDTH, height=GUI_IMAGE_HEIGHT)

prevX, prevY = None, None
cache = None


def drag(e):
    global prevX, prevY, cache
    if prevX is not None and prevY is not None:
        canvas.delete('all')
        paint.line([prevX, prevY, e.x, e.y], fill='black', width=7)
        cache = ImageTk.PhotoImage(image)
        canvas.create_image((0, 0), image=cache, anchor=tkinter.NW)
    prevX = e.x
    prevY = e.y


def reset(e):
    global prevX, prevY
    prevX = None
    prevY = None


canvas.bind('<B1-Motion>', drag)
canvas.bind('<ButtonRelease-1>', reset)


def recognize():
    image_array = np.array(image.resize(size=(mnist_read.IMAGE_WIDTH, mnist_read.IMAGE_HEIGHT)), dtype=np.float32) * 255

    output = session.run(out, feed_dict={in_x: image_array})
    best_prediction = np.argmax(output)
    output[0][best_prediction] = -100
    second_best_prediction = np.argmax(output)

    canvas.create_text(15, 15, text=best_prediction, fill='blue')
    canvas.create_text(15, 30, text=second_best_prediction, fill='red')


def clear():
    canvas.delete('all')
    paint.rectangle([0, 0, GUI_IMAGE_WIDTH, GUI_IMAGE_HEIGHT], fill='white')
    canvas.create_image(0, 0, anchor=tkinter.NW, image=ImageTk.PhotoImage(image))


recognize_button = tkinter.Button(window, text='Recognize', command=recognize)
clear_button = tkinter.Button(window, text='Clear', command=clear)

canvas.pack()
recognize_button.pack()
clear_button.pack()
window.mainloop()

session.close()
