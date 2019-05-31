from fastai import *
from fastai.vision import *
import fastai
import yaml
import sys
from io import BytesIO
from typing import List, Dict, Union, ByteString, Any
import os
import flask
from flask import Flask
import requests
import torch
import json
import torch.nn as nn
import torch.nn.functional as F

with open("src/config.yaml", 'r') as stream:
    APP_CONFIG = yaml.load(stream)

app = Flask(__name__)

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 48, kernel_size=3)
    self.conv2 = nn.Conv2d(48, 96, kernel_size=3)
    self.conv2_drop = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(96 * 5 * 5, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 49)

    # Batch Normalization
    self.bn2d_conv1 = nn.BatchNorm2d(48)
    self.bn2d_conv2 = nn.BatchNorm2d(96)
    self.bn1d_fc1 = nn.BatchNorm1d(256)
    self.bn1d_fc2 = nn.BatchNorm1d(128)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))    # (48, 13, 13)
    x = self.bn2d_conv1(x)
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))   # (96, 5, 5)
    x = self.bn2d_conv2(x)
    x = x.view(-1, x.size(1) * x.size(2) * x.size(3)) #(2400)
    x = F.relu(self.fc1(x))
    x = self.bn1d_fc1(x)
#     x = F.dropout(x, training=self.training)
    x =  F.relu(self.fc2(x))
    x = self.bn1d_fc2(x)
#     x = F.dropout(x, training=self.training)
    x = self.fc3(x)
    x = F.log_softmax(x, dim=-1)
    return x


# def load_model(path=".", model_name="model_kuzushiji2.pkl"):
def load_model(path=".", model_name="model.pkl"):
    # learn = load_learner(path, fname=model_name)
    learn = torch.load("./models/" + model_name, map_location='cpu')
    return learn

def load_image_url(url: str) -> Image:
    response = requests.get(url)
    img = open_image(BytesIO(response.content))
    return img


def load_image_bytes(raw_bytes: ByteString) -> Image:
    img = open_image(BytesIO(raw_bytes))
    return img


def predict(img, n: int = 3) -> Dict[str, Union[str, List]]:
    pred_class, pred_idx, outputs = model.predict(img)
    # print(vars(outputs))
    pred_probs = outputs / sum(outputs)
    pred_probs = pred_probs.tolist()
    predictions = []
    for image_class, output, prob in zip(model.data.classes, outputs.tolist(), pred_probs):
        output = round(output, 1)
        prob = round(prob, 2)
        predictions.append(
            {"class": image_class.replace("_", " "), "output": output, "prob": prob}
        )

    predictions = sorted(predictions, key=lambda x: x["output"], reverse=True)
    predictions = predictions[0:n]
    return {"class": str(pred_class), "predictions": predictions}


@app.route('/api/classify', methods=['POST', 'GET'])
def upload_file():
    if flask.request.method == 'GET':
        url = flask.request.args.get("url")
        img = load_image_url(url)
    else:
        bytes = flask.request.files['file'].read()
        img = load_image_bytes(bytes)
    res = predict(img)
    return flask.jsonify(res)


@app.route('/api/classes', methods=['GET'])
def classes():
    classes = sorted(model.data.classes)
    return flask.jsonify(classes)


@app.route('/ping', methods=['GET'])
def ping():
    return "pong"


@app.route('/config')
def config():
    return flask.jsonify(APP_CONFIG)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"

    response.cache_control.max_age = 0
    return response


@app.route('/<path:path>')
def static_file(path):
    if ".js" in path or ".css" in path:
        return app.send_static_file(path)
    else:
        return app.send_static_file('index.html')


@app.route('/')
def root():
    return app.send_static_file('index.html')



def before_request():
    app.jinja_env.cache = {}


model = load_model('models')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)

    if "prepare" not in sys.argv:
        app.jinja_env.auto_reload = True
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        app.run(debug=False, host='0.0.0.0', port=port)
        # app.run(host='0.0.0.0', port=port)
