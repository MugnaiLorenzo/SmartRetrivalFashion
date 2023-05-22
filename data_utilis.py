import csv
import torchvision.transforms.functional as F
import PIL.Image
import torch
import pickle
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset

from collections import Counter
from fashion_clip.fashion_clip import FashionCLIP, FCLIPDataset
from pathlib import Path
from typing import Optional
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

server_base_path = Path(__file__).absolute().parent.absolute()
dataset_root = Path(__file__).absolute().parent.absolute() / 'data_set'
data_path = Path(__file__).absolute().parent.absolute() / 'data'
image_root = dataset_root / 'images'
images = []


def _createSubSet():
    global subset
    path = dataset_root / "articles.csv"
    articles = pd.read_csv(path, on_bad_lines='skip')
    subset = articles.drop_duplicates("detail_desc").copy()
    subset = subset[~subset["product_group_name"].isin(["Unknown"])]
    subset = subset[subset["detail_desc"].apply(lambda x: 4 < len(str(x).split()) < 40)]
    most_frequent_product_types = [k for k, v in dict(Counter(subset["product_type_name"].tolist())).items() if v > 10]
    subset = subset[subset["product_type_name"].isin(most_frequent_product_types)]
    path_image = dataset_root / 'images'
    for k in subset["article_id"].tolist():
        row = []
        path = str(path_image) + '/0' + str(k)[:2] + '/0' + str(k) + ".jpg"
        if os.path.isfile(path):
            row.append(k)
            row.append(path)
            images.append(row)


def _load_assets():
    _createSubSet()
    global fclip
    fclip = FashionCLIP('fashion-clip')
    global dataset_index_features
    dataset_index_features = torch.load(data_path / 'dataset_index_features.pt')
    global dataset_index_name
    with open(data_path / 'dataset_index_names.pkl', 'rb') as f:
        dataset_index_name = pickle.load(f)


def _get_id_img_from_text(text: str):
    if len(images) == 0:
        _createSubSet()
    # precision = 0
    # # we could batch this operation to make it faster
    # for index, t in enumerate(dataset_index_name):
    #     arr = t.dot(dataset_index_features.T)
    #
    #     best = arr.argsort()[-5:][::-1]
    #
    #     if index in best:
    #         precision += 1

    # print(round(precision / len(dataset_index_name), 2))
    text_embedding = fclip.encode_text([text], 32)[0]
    id_of_matched_object = np.argmax(text_embedding.dot(dataset_index_features.T))
    found_object = subset["article_id"].iloc[id_of_matched_object].tolist()
    print(found_object)
    return found_object


def get_img_from_id(id: str):
    if len(images) == 0:
        _createSubSet()
    for i in images:
        if str(i[0]) == str(id):
            return i[1]
    return None


# check is an image
def is_image_file(filename: str) -> bool:
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


# conversion image to rgb
def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_char_image(name: str):
    with open(dataset_root / 'articles.csv', encoding="utf8") as file:
        csv_reader = csv.DictReader(file, delimiter=',')
        for row in csv_reader:
            if str(row['article_id']) == '0' + str(name):
                return row
    return None


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad it to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int, pad_value: Optional[int] = 0):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        :param pad_value: padding value, 0 is black-pad (zero-pad), 255 is white-pad
        """
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
    """
    Yield a torchvision transform which resize and center crop an image using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :param pad_value: padding value, 0 is black-pad (zero-pad), 255 is white-pad
    :return: torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim, pad_value),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
    ])


class Fashion_Dataset(Dataset):
    def __init__(self, preprocess: callable):
        super().__init__()
        if len(images) == 0:
            _createSubSet()
        self.preprocess = preprocess
        self.dataset_root = dataset_root
        path = dataset_root / "articles.csv"
        paths = []
        self.img_labels = pd.read_csv(path, on_bad_lines='skip')
        path_image = dataset_root / 'images'
        for k in self.img_labels["article_id"].tolist():
            path = str(path_image) + '/0' + str(k)[:2] + '/0' + str(k) + ".jpg"
            if os.path.isfile(path):
                paths.append(path)
        self.image_paths = path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = str(self.image_paths[index])
        try:
            image = self.preprocess(PIL.Image.open(self.image_paths[index]))
        except Exception as e:
            print(f"Exception occured: {e}")
            return None
        return image, image_path
