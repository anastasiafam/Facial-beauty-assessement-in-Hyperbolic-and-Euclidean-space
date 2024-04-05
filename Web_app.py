from __future__ import division, print_function

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


import sys
import os
import glob
import re

import argparse
from PIL import Image
import torch
from torchvision import transforms



def image_loader(path_image, loader,device):
    
    """load image, returns cuda tensor"""
    image = Image.open(path_image)
    image = loader(image).float()
    image = image.view(1, *image.shape)

    return image.to(device)  #assumes that you're using GPU


def predict(path_model, image, device):
    
    model = torch.load(path_model, map_location=torch.device(device))
    model = model.to(device)
    model.eval()
    result = model(image)
    
    return result


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        
        selected_option = request.form['selected_option']
        print(selected_option)
        
        image_path = file_path
        print(image_path)

        if (selected_option=='model_evk'):
            model_path="/home/anastasia/MEBeauty-database/pytorch_trained_models/model_20240402-190026.pht"
        elif (selected_option=='model_giper'):
            model_path="/home/anastasia/MEBeauty-database/pytorch_trained_models/model_20240402-190026.pht"
        
        imsize = (256, 256)
        loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(device)
        image = image_loader(image_path, loader,device)
        print("ok")
        score = predict(model_path, image, device)
        print(score)
        result = str(score.item())
        print("sending")
        return result
    return "RAISE ERROR"
app.run()
