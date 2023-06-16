import math
import os
import random
import data_utilis
import PIL.Image

from flask import Flask, render_template, send_file, request, url_for, redirect
from io import BytesIO
from data_utilis import *
from data_utilis import targetpad_resize

app = Flask(__name__)
upload = image_root
app.config['UPLOAD'] = upload
app.config['UPLOAD_TEMP'] = server_base_path / "static" / "Image" / "temporary_file"


@app.route('/')
@app.route('/home')
def home():  # put application's code here
    load()
    images = get_random_images(12)
    return render_template('base.html', names=images, active="Home", cols=get_collections_name())


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def search():  # put application's code here
    load()
    if request.method == 'POST':
        search = request.form['search']
        collection = request.form['collection']
        image = request.files['image']
        if not search == "":
            imgs = retrival_from_text(search, collection)
            return render_template('result.html', imgs=imgs, search=search, cols=get_collections_name())
        else:
            img = PIL.Image.open(image)
            imgs = get_label_from_image(img, collection)
            return render_template('result.html', imgs=imgs, search=search, cols=get_collections_name())


@app.route('/get_image/<string:image_name>/<int:collection>')
@app.route('/get_image/<string:image_name>/<int:collection>/<int:dim>')
def get_image(image_name: str, collection: str, dim: Optional[int] = None):
    if not is_load():
        load()
    path_image = str(image_root) + "/" + "collection_" + str(collection) + "/" + image_name + ".jpg"
    if dim:
        transform = targetpad_resize(1.25, int(dim), 255)
        pil_image = transform(PIL.Image.open(path_image))
    else:
        pil_image = PIL.Image.open(path_image)
    img_io = BytesIO()
    pil_image.save(img_io, 'JPEG', quality=80)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


@app.route('/char_image/<string:image_name>/<string:collection>')
def char_image(image_name: str, collection: str):
    if not is_load():
        load()
    param = setParam(image_name, int(collection))
    return render_template('feature.html', id=image_name, collection=collection, param=param,
                           cols=get_collections_name())


@app.route('/add/')
def add():
    return render_template('add_collection.html', page=1)


@app.route('/add/', methods=['GET', 'POST'])
def add_post():
    global i
    names = []
    images = []
    len = 0
    if request.method == 'POST':
        name = request.form['name']
        i = request.files.getlist("image")
        for j in i:
            len = len + 1
            names.append(j.filename)
            j.save(os.path.join(app.config['UPLOAD_TEMP'], j.filename))
            path = "Image/temporary_file/" + j.filename
            images.append(path)
    return render_template('add_collection.html', page=2, names=names, image=images, len=len)


@app.route('/modify')
def modify():
    if not is_load():
        load()
    cs = get_collection_for_modify()
    return render_template('modify.html', active="Modify", collections=cs, cols=get_collections_name())


@app.route('/collection/<col>')
@app.route('/collection/<col>/<page>')
def collection(col, page: Optional[int] = 1, operation: Optional[str] = "s"):
    if not is_load():
        load()
    catalog, n_col, col_name = get_image_from_collection(int(col))
    n_page = n_col / 15
    n_page = str(int(n_page))
    if str(page) == n_page:
        ls = int(n_col)
    else:
        ls = int(page) * 15 + 15
    li = int(page) * 15
    page_p = int(page) - 1
    page_n = int(page) + 1
    return render_template('modify_collection.html', active="Modify", l_i=li, l_s=ls, n_page=n_page, catalog=catalog,
                           n_col=n_col, c_id=col, page=page, page_n=page_n, page_p=page_p, col_name=col_name)


if __name__ == '__main__':
    app.run()
