#!/usr/bin/env python
__author__ = 'Aaron Woodhouse'

"""
Application for skin cancer prediction.
"""

# Imports #
import sys
import os

from tkinter import ttk
from tkinter import *

DIR = os.path.dirname(os.path.realpath(__file__))
#PARENT = os.path.dirname(DIR)
sys.path.append(DIR)

from models import *
import resizer
import CancerPrediction

class Application():
    """Skin cancer predictor application."""
    
    def __init__(self):
        """Initialize Application.

        Attributes
        ----------
        p1 : :PhotoImage:
            Icon image for windows.
        e1
            Metadata path entry field.
        e2
            Image folder path entry field.
        combo
            Model selection dropdown.

        """
        window = Tk()
        window.geometry("800x210")
        window.title("Cancer Predictor")
        self.p1 = PhotoImage(file = os.path.join(DIR, 'icon.png'))
        window.iconphoto(False, self.p1)

        Label(window, text='Data path').grid(sticky = W, row=0, padx=10, pady=10)
        Label(window, text='Image folder path').grid(sticky = W, row=1, padx=10, pady=10)
        Label(window, text='Resize images').grid(sticky = W, row=2, padx=10, pady=10)
        Label(window, text='Model').grid(sticky = W, row=3, padx=10, pady=10)

        self.e1 = Entry(window, width=90)
        self.e2 = Entry(window, width=90)

        # default path set to deployment data
        self.e1.insert(END, os.path.join(DIR, 'Data\deployment_data.csv'))
        self.e2.insert(END, os.path.join(DIR, 'Data\Deployment_Images'))

        self.combo = ttk.Combobox(
            state="readonly",
            values = ['Linear', 'MLP', 'CNN', 'DCNN','CM1','CM2','CM3','DenseNet']
        )

        resize_button = ttk.Button(text="Resize", command=self.resize)
        deploy_button = ttk.Button(text="Predict", command=self.deploy)


        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        resize_button.grid(sticky = W, row=2, column=1)
        self.combo.grid(sticky = W, row=3, column=1)
        deploy_button.grid(sticky = W, row=4, column=1)

        mainloop()
    
    def _to_filename(self, model: str):
        """Take model name, and get filename.
    
        Parameters
        ----------
        model : str
            Model name.
    
        Returns
        -------
        str
            Model file name.
    
        """
        
        files = {
                'Linear': 'linear_model.pt',
                'MLP': 'mlp_model.pt',
                'CNN': 'cnn_model.pt',
                'DCNN': 'dcnn_model.pt',
                'CM1': 'cm1_model.pt',
                'CM2': 'cm2_model.pt',
                'CM3': 'cm3_model.pt',
                'DenseNet': 'dense_net_model.pt'
                }
        
        return files[model]
    
    def deploy(self):
        """Deploy specified model on dataset.
        
        Raises
        ------
        SyntaxError
            If data file given is not a path.
            If image folder given is not a path.
        AttributeError
            If model is not chosen.
        
        """
        
        if not os.path.isfile(self.e1.get()):
            raise SyntaxError("Data file given is not a path")
            
        if not os.path.isdir(self.e2.get()):
            raise SyntaxError("Image folder given is not a path")
            
        if not self.combo.get():
            raise AttributeError("Model not chosen")
        
        print("Deploying...")
        
        data = self.e1.get()
        imgs = self.e2.get()
        model = self.combo.get()
        model = self._to_filename(model)
        results = CancerPrediction.deploy(data, imgs, model)
        
        result_window = Toplevel()
        result_window.title('Results')
        result_window.iconphoto(False, self.p1)
        
        msg = Message(result_window, text = results)
        msg.pack( )
        print("Done")
    
    def resize(self):
        """Resize images in specified image folder path.
        
        Raises
        ------
        SyntaxError
            If image folder given is not a path.
        
        """
        
        if not os.path.isdir(self.e2.get()):
            raise SyntaxError("Image folder given is not a path")
        
        print("Resizing...")
        resizer.resize(self.e2.get())
        print("Done")
        
def main():
    app = Application()

if __name__=='__main__':
    main()










