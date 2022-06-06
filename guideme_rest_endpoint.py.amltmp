


import argparse
import io

import torch
from flask import Flask, request
from PIL import Image
import os 

app = Flask(__name__)

DETECTION_URL = "/v1/guideme"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if request.method != "POST":
        return

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        results = model(im, size=640)  
        return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=4444, type=int, help="port number")
    opt = parser.parse_args()

   
    model_name = "yolov5n"
    model = torch.hub.load(os.getcwd(), 'custom', source='local', path = model_name, force_reload = False)
    app.run(host="0.0.0.0", port=opt.port)