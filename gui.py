from matplotlib import pyplot as plt
import numpy as np
import ttkbootstrap as ttk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import cv2 as cv
from PIL import Image

class GUI(ttk.Window):
    def __init__(self):
        # setup window
        super().__init__(themename='darkly')
        self.title('Image Classifier')
        self.geometry('400x300')
        self.resizable(height=False, width=False)

        # load model 
        self.model = load_model('image_class.keras')
        self.class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

        # create widgets
        self.create_widgets()

        # run
        self.mainloop()

    def create_widgets(self):
        self.button_explore = ttk.Button(self, 
                                text = "Browse Files",
                                command = self.browseFiles).place(relx=0.5, rely=0.3, anchor='center')
        
        self.label_file_explorer = ttk.Label(self, 
                            text = "Select Image to Classify")
        
        self.label_file_explorer.place(relx=0.5, rely=0.6, anchor='center')
        
    def browseFiles(self):
        filename = filedialog.askopenfilename(initialdir = "/",
                                            title = "Select a File",
                                            filetypes = (
                                                        ("all files",
                                                            "*.*"),
                                                        ("Text files",
                                                            "*.txt*")))
        
        self.classify(filename)

    def classify(self, url):
        try:
            with Image.open(url) as img:
                img_resized = img.resize((32, 32))
                
                img_resized = np.array(img_resized)
                
                img_resized = cv.cvtColor(img_resized, cv.COLOR_RGB2BGR)
                
                plt.imshow(img_resized[:, :, ::-1], cmap=plt.cm.binary)
                
                prediction = self.model.predict(np.array([img_resized]) / 255.0)
                index = np.argmax(prediction)
                result = self.class_names[index]
                
                self.label_file_explorer.configure(text=result)
                
        except Exception as e:
            print(f"Error: {e}")
               
g = GUI()
g