import tkinter as tk
import tensorflow as tf
from mnist_model import model_fn
import numpy as np
from threading import Thread
from time import sleep
from PIL import Image
import io

SIZE=400

class PredictDraw:
    def __init__(self):
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.root.destroy())
        self.tk = tk
        self.container = tk.Frame(self.root, width=SIZE, height=SIZE, padx=10, pady=10, bg='white')
        self.container.winfo_toplevel().title('MNIST DRAW')
        self.container.pack(side='top', fill='both', expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        self.prediction_label = tk.Label(self.container, text='NO PREDICTION', font=20)
        self.prediction_label.pack(side=tk.BOTTOM)
        
        self.clear_button = tk.Button(self.container, text='clear', command=self.clear_canvas, relief=tk.GROOVE)
        self.clear_button.pack(side=tk.BOTTOM)
        
        self.prediction =None
        self.mouseon = False
        self.img_data = None
        self.init_tensorflow()
        self.create_canvas()
        self.thread = None
        
        self.root.mainloop()

    def create_canvas(self):
            self.canvas = tk.Canvas(self.container, width = 28*10, height = 28*10)
            self.canvas.bind("<Button>", self.mouse_click)
            self.canvas.bind("<ButtonRelease>", self.mouse_release)
            self.canvas.bind("<Motion>", self.mouse_moved)
            self.canvas.pack()
            self.clear_canvas()

            self.thread = Thread(target=self.predict_in_background)
            self.thread.start()
    
    def clear_canvas(self):
        self.canvas.create_rectangle(0,0, self.canvas['width'], self.canvas['height'], fill="black", outline='black')
        
    
    def mouse_moved(self, e):
        if self.mouseon:
            self.canvas.create_oval(e.x-7, e.y-7, e.x+7, e.y+7, fill="white", outline='white')
            
    def mouse_release(self, e):
        if e.num==1:
            self.mouseon = False
    
    def mouse_click(self, e):
        if e.num==1:
            self.mouseon = True
            self.canvas.create_oval(e.x-7 , e.y-7, e.x+7, e.y+7, fill="white", outline='white')

    def predict_in_background(self):
        while True:
            #1/3 of a second ( not in millis!)
            sleep(1/3)
            # GET CANVAS IMG
            self.canvas.update()
            img = self.canvas.postscript(colormode='color')
            # CONVERT from RGB to 8 bit 0-255 floating point pixel
            img = Image.open(io.BytesIO(img.encode('utf-8'))).convert('L')
            
            # Greatly converts img to be predictable (trained imgs was 28x28)
            img.thumbnail((28,28))

            # The model was trained for 0-1 floats (grayscale)
            self.img_data = [ p/255. for p in img.getdata()]
            self.img_data = np.array(self.img_data)

            # PREDICT
            prediction = self.predict(self.img_data)

            # Update prediction tag
            self.prediction_label['text'] = 'Prediction : {}'.format(str(prediction['classes']))
        
    # Tensorflow operations
    def init_tensorflow(self):
        self.model = tf.estimator.Estimator(model_fn=model_fn, model_dir='./data/MNIST/model')
        
    def predict(self, img_data):
        # As estimator works with 32 bits multidimensional arrays
        img_input = np.reshape(img_data, (1,28,28,1)).astype(np.float32)
        
        # `list` makes the model.predict generator that is returned  eval
        return list(self.model.predict(tf.estimator.inputs.numpy_input_fn(
            x={'x': img_input},
            shuffle=False)))[0] # the first prediction (cause img_input len == 1)

PredictDraw()
