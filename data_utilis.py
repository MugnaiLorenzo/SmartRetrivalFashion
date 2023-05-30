import csv
import pickle
from pathlib import Path
from typing import Optional

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms.functional as F
import PIL.Image
import pandas as pd
import os
import numpy as np

from collections import Counter

server_base_path = Path(__file__).absolute().parent.absolute()
dataset_root = Path(__file__).absolute().parent.absolute() / 'data_for_fashion_clip'
image_root = dataset_root / 'images'
data_path = Path(__file__).absolute().parent.absolute() / 'data'
image_id = []


# read cvs file
def read_cvs():
    with open(dataset_root / 'articles.csv', encoding="utf8") as file:
        csv_reader = csv.DictReader(file, delimiter=',')
        line_count = 0
        global catalog
        catalog = []
        for row in csv_reader:
            if line_count != 0:
                catalog.append(
                    {'id': row['article_id'], 'image': get_url(row['article_id']), 'caption': row['detail_desc']})
                image_id.append(row['article_id'])
            line_count += 1


def setSubset():
    global fclip
    global subset
    # dataset = FCLIPDataset(name="mydataset", image_source_path=str(image_root), image_source_type='local', catalog=catalog)
    # fclip = FashionCLIP('fashion-clip', dataset)
    with open(data_path / 'f_clip.pkl', 'rb') as c:
        fclip = pickle.load(c)
    path = dataset_root / "articles.csv"
    articles = pd.read_csv(path, on_bad_lines='skip')
    # drop items that have the same description
    subset = articles.drop_duplicates("detail_desc").copy()
    # remove items of unkown category
    subset = subset[~subset["product_group_name"].isin(["Unknown"])]
    # FashionCLIP has a limit of 77 tokens, let's play it safe and drop things with more than 40 tokens
    subset = subset[subset["detail_desc"].apply(lambda x: 4 < len(str(x).split()) < 40)]
    # We also drop products types that do not occur very frequently in this subset of data
    most_frequent_product_types = [k for k, v in dict(Counter(subset["product_type_name"].tolist())).items() if v > 10]
    subset = subset[subset["product_type_name"].isin(most_frequent_product_types)]
    path_image = dataset_root / 'images'
    global images
    images = []
    for k in subset["article_id"].tolist():
        if os.path.isfile(str(path_image) + "/" + str(k) + ".jpg"):
            images.append(str(path_image) + "/" + str(k) + ".jpg")


def load():
    read_cvs()
    setSubset()
    global dataset_index_features
    dataset_index_features = torch.load(data_path / 'dataset_index_features.pt')
    global dataset_index_name
    with open(data_path / 'dataset_index_names.pkl', 'rb') as f:
        dataset_index_name = pickle.load(f)
    retrival_from_text('jeans')


# check is an image
def is_image_file(filename: str) -> bool:
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


# conversion image to rgb
def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_url(name: str):
    return str(image_root) + "/" + name + ".jpg"


def get_char_image(name: str):
    with open(dataset_root / 'articles.csv', encoding="utf8") as file:
        csv_reader = csv.DictReader(file, delimiter=',')
        for row in csv_reader:
            if str(row['article_id']) == str(name):
                return row
    return None


def get_id_from_text(text: str):
    text_embedding = fclip.encode_text([text], 32)[0]
    id_of_matched_object = np.argmax(text_embedding.dot(dataset_index_features.T))
    found_object = subset["article_id"].iloc[id_of_matched_object].tolist()
    return found_object


def retrival_from_text(text: str):
    imgs = []
    r = fclip.retrieval([text])[0]
    for i in r:
        imgs.append(catalog[i])
    return imgs


class TargetPad:
    # Pad the image if its aspect ratio is above a target ratio.
    # Pad it to match such target ratio
    def __init__(self, target_ratio: float, size: int, pad_value: Optional[int] = 0):
        #:param target_ratio: target ratio
        #:param size: preprocessing output dimension
        #:param pad_value: padding value, 0 is black-pad (zero-pad), 255 is white-pad
        self.size = size
        self.target_ratio = target_ratio
        self.pad_value = pad_value

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, self.pad_value, 'constant')


def targetpad_resize(target_ratio: float, dim: int, pad_value: int):
    # Yield a torchvision transform which resize and center crop an image using TargetPad pad
    #:param target_ratio: target ratio for TargetPad
    #:param dim: image output dimension
    #:param pad_value: padding value, 0 is black-pad (zero-pad), 255 is white-pad
    #:return: torchvision Compose transform
    return Compose([
        TargetPad(target_ratio, dim, pad_value),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
    ])
