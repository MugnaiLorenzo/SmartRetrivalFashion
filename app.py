# import pickle
import pickle

import torch
import random

import data_utilis
import numpy as np
import PIL.Image

# import pandas as pd
from flask import Flask, render_template, send_file, request, url_for, redirect
from io import BytesIO
from data_utilis import *
from data_utilis import targetpad_resize

# from fashion_clip.fashion_clip import FashionCLIP

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():  # put application's code here
    load()
    search = request.args.get('search')
    active = "Home"
    if search is None:
        random_indexes = random.sample(range(len(data_utilis.image_id)), k=15)
        names = np.array(data_utilis.image_id)[random_indexes].tolist()
        return render_template('base.html', names=names, active="Home")
    else:
        image_name = get_id_from_text(search)
        param = setParam(image_name)
        return render_template('feature.html', id=image_name, param=param)


@app.route('/get_image/<string:image_name>')
@app.route('/get_image/<string:image_name>/<int:dim>')
def get_image(image_name: str, dim: Optional[int] = None):
    name = image_name + ".jpg"
    image_path = image_root / name
    if dim:
        transform = targetpad_resize(1.25, int(dim), 255)
        pil_image = transform(PIL.Image.open(image_path))
    else:
        pil_image = PIL.Image.open(image_path)
    img_io = BytesIO()
    pil_image.save(img_io, 'JPEG', quality=80)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


@app.route('/char_image/<string:image_name>')
def char_image(image_name: str):
    param = setParam(image_name)
    return render_template('feature.html', id=image_name, param=param)


@app.route('/add/', methods=['GET'])
def add():
    name = request.args.get('name')
    description = request.args.get('description')
    type = request.args.get('type')
    group = request.args.get('group')
    colour = request.args.get('colour')
    image = request.args.get('image')
    print(name, description, type, group, colour, image)
    return redirect(url_for('home'))


def setParam(image_name: str):
    param = []
    char = data_utilis.get_char_image(image_name)
    param.append(char['prod_name'])
    param.append(char['detail_desc'])
    param.append(char['product_type_name'])
    param.append(char['product_group_name'])
    param.append(char['colour_group_name'])
    return param


@app.route('/modify')
def modify():
    active = "Modify"
    return render_template('modify.html', active=active)


if __name__ == '__main__':
    app.run()
