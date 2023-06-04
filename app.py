import random
import data_utilis
import PIL.Image

from flask import Flask, render_template, send_file, request, url_for, redirect
from io import BytesIO
from data_utilis import *
from data_utilis import targetpad_resize

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():  # put application's code here
    load()
    images = []
    for i in range(0, 15):
        row = []
        random_c = random.sample(range(data_utilis.get_cols().get_len()), 1)
        random_c = random_c[0] + 1
        row.append(random_c)
        row.append(data_utilis.get_cols().get_collection_from_id(random_c).get_random_image())
        images.append(row)
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
    if not get_cols():
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
    load()
    param = setParam(image_name, collection)
    return render_template('feature.html', id=image_name, collection=collection, param=param,
                           cols=get_collections_name())


@app.route('/add/', methods=['GET', 'POST'])
def add():
    name = request.form['name']
    description = request.form['description']
    type = request.form['type']
    group = request.form['group']
    colour = request.form['colour']
    image = request.files['image']
    print(name, description, type, group, colour, image.filename)
    return redirect(url_for('home'))


@app.route('/modify')
def modify():
    load()
    return render_template('modify.html', active="Modify", cols=get_collections_name())


if __name__ == '__main__':
    app.run()
