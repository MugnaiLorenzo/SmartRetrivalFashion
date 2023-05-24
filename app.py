# import pickle
import torch
import random

import data_utilis
import numpy as np
import PIL.Image

# import pandas as pd
from flask import Flask, render_template, send_file, request
from io import BytesIO
from data_utilis import *
from data_utilis import targetpad_resize
# from fashion_clip.fashion_clip import FashionCLIP

app = Flask(__name__)


@app.before_request
def _load_assets():
    data_utilis.read_cvs()
    # global fclip
    # fclip = FashionCLIP('fashion-clip')
    global dataset_index_features
    dataset_index_features = torch.load(data_path / 'dataset_index_features.pt')
    global dataset_index_name
    with open(data_path / 'dataset_index_name.pkl', 'rb') as f:
        dataset_index_name = pickle.load(f)
    #
    # text_embedding = fclip.encode_text(["trousers"], 32)[0]
    # id_of_matched_object = np.argmax(text_embedding.dot(dataset_index_name.T))
    # # found_object = subset["productDisplayName"].iloc[id_of_matched_object].tolist()
    # path = dataset_root / "styles.csv"
    # articles = pd.read_csv(path, on_bad_lines='skip')
    # subset = articles.drop_duplicates("productDisplayName").copy()
    # # remove items of unkown category
    # subset = subset[~subset["masterCategory"].isin(["Unknown"])]
    # # FashionCLIP has a limit of 77 tokens, let's play it safe and drop things with more than 40 tokens
    # subset = subset[subset["productDisplayName"].apply(lambda x: 4 < len(str(x).split()) < 40)]
    # # We also drop products types that do not occur very frequently in this subset of data
    # found_object = subset["productDisplayName"].iloc[id_of_matched_object]
    # print(found_object)



@app.route('/')
def start():  # put application's code here
    search = request.args.get('search')
    if search is None:
        random_indexes = random.sample(range(len(data_utilis.image_id)), k=15)
        names = np.array(data_utilis.image_id)[random_indexes].tolist()
        return render_template('base.html', names=names)
    else:
        return render_template('result.html')


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
    char = data_utilis.get_char_image(image_name)
    print(char['gender'], char['articleType'], char['baseColour'], char['year'], char['productDisplayName'])
    gender = char['gender']
    articleType = char['articleType']
    baseColour = char['baseColour']
    year = char['year']
    productDisplayName = char['productDisplayName']
    return render_template('feature.html', name=image_name, gender=gender, articleType=articleType,
                           baseColour=baseColour, year=year, productDisplayName=productDisplayName)


@app.route('/modify')
def modify():
    return render_template('modify.html')


if __name__ == '__main__':
    app.run()
