import json
import math
import os
import random
import data_utilis
import PIL.Image
import shutil

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
def home():
    load()
    images = get_random_images(6)
    return render_template('base.html', names=images, active="Home", cols=get_collections_name())


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def search():
    if not is_load():
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
    global col_name
    names = []
    images = []
    len = 0
    if request.method == 'POST':
        col_name = request.form['name']
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
    n = 18
    if not is_load():
        load()
    catalog, n_col, col_name = get_image_from_collection(int(col))
    if n_col < n:
        n = n_col
    n_page = n_col / n
    n_page = str(int(n_page))
    if str(page) == n_page:
        ls = int(n_col)
    else:
        ls = int(page) * n + n
    li = int(page) * n - n
    page_p = int(page) - 1
    page_n = int(page) + 1
    return render_template('modify_collection.html', active="Modify", l_i=li, l_s=ls, n_page=int(n_page),
                           catalog=catalog, n_col=n_col, c_id=col, page=int(page), page_n=page_n, page_p=page_p,
                           col_name=col_name)


@app.route('/load', methods=['GET', 'POST'])
def load_collection():
    n = get_len_of_collection() + 1
    par = []
    images = []
    if request.method == 'POST':
        cn = col_name.split(" ")
        c_name = ""
        for c in cn:
            c_name = c_name + c
        path = str(image_root) + "/" + "collection_" + str(n)
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)
        for j in i:
            name = request.form["name" + j.filename]
            description = request.form["description" + j.filename]
            type = request.form["type" + j.filename]
            group = request.form["group" + j.filename]
            colour = request.form["colour" + j.filename]
            id = j.filename.split(".")
            row = {
                "article_id": id[0],
                "prod_name": name,
                "product_type_name": type,
                "product_group_name": group,
                "colour_group_name": colour,
                "detail_desc": description
            }
            par.append(row)
            shutil.move(server_base_path / "static" / "Image" / "temporary_file" / j.filename, path)
            images.append(path + "/" + str(j.filename))
        json_path = str(data_utilis.metadata_path) + "/collection_" + str(n) + ".json"
        with open(json_path, 'w') as outfile:
            json.dump(par, outfile)
        fclip_path = data_utilis.set_dataset_json(c_name)
        data_utilis.embedding_image(images, fclip_path)
    update_chroma()
    return redirect(url_for('home'))


@app.route('/delete_collection/<col>')
def delete_collection(col):
    if not is_load():
        load()
    data_utilis.delete_col(col)
    update_chroma()
    return redirect(url_for('home'))


@app.route('/add_image_at_collection', methods=['GET', 'POST'])
def add_image_at_collection():
    global i
    global col_id
    names = []
    images = []
    len = 0
    path = str(server_base_path) + "/static/Image/temporary_file"
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)
    if request.method == 'POST':
        col_id = request.form['name']
        i = request.files.getlist("image")
        for j in i:
            len = len + 1
            names.append(j.filename)
            j.save(os.path.join(app.config['UPLOAD_TEMP'], j.filename))
            path = "Image/temporary_file/" + j.filename
            images.append(path)
    return render_template('add_metadata.html', names=names, image=images, len=len)


@app.route('/load_image', methods=['GET', 'POST'])
def load_image():
    if not is_load():
        load()
    images = []
    if request.method == 'POST':
        path = str(image_root) + "/" + "collection_" + col_id
        collection = get_collection_from_index(int(col_id))
        par = collection.peek()['metadatas']
        for p in par:
            images.append(path + "/" + str(p['article_id']) + ".jpg")
        for j in i:
            name = request.form["name" + j.filename]
            description = request.form["description" + j.filename]
            type = request.form["type" + j.filename]
            group = request.form["group" + j.filename]
            colour = request.form["colour" + j.filename]
            id = j.filename.split(".")
            row = {
                "article_id": id[0],
                "prod_name": name,
                "product_type_name": type,
                "product_group_name": group,
                "colour_group_name": colour,
                "detail_desc": description
            }
            par.append(row)
            shutil.move(server_base_path / "static" / "Image" / "temporary_file" / j.filename, path)
            images.append(path + "/" + str(j.filename))
        json_path = str(data_utilis.metadata_path) + "/collection_" + str(col_id) + ".json"
        with open(json_path, 'w') as outfile:
            json.dump(par, outfile)
        n = "f_clip_" + str(col_id) + ".pkl"
        fclip_path = "dataset\\Fclip\\" + n
        data_utilis.embedding_image(images, fclip_path)
    update_chroma()
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run()
