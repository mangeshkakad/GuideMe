


import torch
import os
import numpy as np
import pickle
from azureml.core.model import Model
from azureml.core import Workspace
import gtts
from googletrans import Translator
import sys
import argparse
import io

import torch
from flask import Flask, request
from PIL import Image
import os 


import threading
import requests
import time
import json
from multiprocessing import Process, Queue
import ast

app = Flask(__name__)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

GUIDEME_URL = "/v1/guideme"
HOST = "0.0.0.0"
PORT = 4444

@app.route(GUIDEME_URL, methods=["POST"])
def predict():
    if request.method != "POST":
        return

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        results = model(im, size=640)  
        return (results.pandas().xyxy[0]['name']).to_json()

def guideme_rest_endpoint(q,model_name):
    global model
    model_name = model_name
    model = load_model(model_name)
    app.run(host=HOST, port=PORT)

# Model

def load_model(model_name):
    ws = Workspace(subscription_id="d7a3edde-9913-4bbd-ba4b-7ff57ffa209e",
              resource_group="odl-hackathon-585253",
              workspace_name="mk_demo")
    model_name = "yolov5n"
    model = torch.hub.load(os.getcwd()+"/yolov5/", 'custom', source='local', path = model_name, force_reload = False)
    return model
    

def register_model(filename):
    ws = Workspace(subscription_id="d7a3edde-9913-4bbd-ba4b-7ff57ffa209e",
               resource_group="odl-hackathon-585253",
               workspace_name="mk_demo")
    
    model = Model.register(ws, model_name="guide_me", model_path=filename)

def call_guideme_endpoint(image_name):
    endpoints_url = 'http://'+HOST+':'+str(PORT)+GUIDEME_URL
    img = {'image': open(image_name,'rb')}
    resp = requests.post(endpoints_url, files=img)
    return (resp.json())

def guideme_image_scan(image_name):
    results = call_guideme_endpoint(image_name)
    values, counts = np.unique(list(results.values()), return_counts=True)
    if len(counts):
        guide_text = "There are"
        for val , count in zip(values,counts):
            guide_text = guide_text + f' {count} {val} and'

        guide_text = guide_text[:-3]
        guide_text = f'{guide_text}in front of you.Please be careful.'
    else:
        guide_text = 'Looks like nothing around you.Please be careful but you can progress.'
        
    return guide_text

def guideme_voice(guideme_text,img,language='en'):
    tts = gtts.gTTS(guideme_text,lang=language)
    tts.save(img.split(".")[0]+".mp3")

def guideme_translate(guideme_text,language='en'):
    translator = Translator()
    trans = translator.translate(guideme_text,dest=language)
    return trans.text

def main(args):
    print (args)
    queue = Queue()
    guide_meprocess = Process(target=guideme_rest_endpoint, args=(queue, 'Guide_me'))
    guide_meprocess.start()
    time.sleep(8)
    for img,lang in zip(ast.literal_eval(args[0]),ast.literal_eval(args[1])):
        start_time = time.time()
        guideme_text = guideme_image_scan(img)
        ####print (guideme_text)
        guideme_text_translate = guideme_translate(guideme_text,lang)
        print(f'Message : {guideme_text_translate}')
        guideme_voice(guideme_text_translate,img,lang)
        print("Processing Duration : --- %s miliseconds ---" % ((time.time() - start_time)*1000))





if __name__ == "__main__":
   main(sys.argv[1:])

