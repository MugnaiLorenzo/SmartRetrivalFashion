import csv
import pickle
import torchvision.transforms.functional as F
import PIL.Image

from pathlib import Path
from typing import Optional
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from collection import Collections, Collection

server_base_path = Path(__file__).absolute().parent.absolute()
dataset_root = Path(__file__).absolute().parent.absolute() / 'dataset'
image_root = dataset_root / 'Images'
data_path = Path(__file__).absolute().parent.absolute() / 'data'


def load():
    global cols
    cols = Collections()
    with open(dataset_root / 'dataset.csv', encoding="utf8") as f:
        dataset_reader = csv.DictReader(f, delimiter=',')
        for r in dataset_reader:
            with open(server_base_path / r['metadata_path'], encoding="utf8") as file:
                csv_reader = csv.DictReader(file, delimiter=',')
                catalog = []
                for row in csv_reader:
                    catalog.append({'id': row['article_id'], 'image': get_url(r['image_path'], row['article_id']),
                                    'caption': row['detail_desc']})
                p = server_base_path / r['fclip_path']
                with open(p, 'rb') as c:
                    fclip = pickle.load(c)
                fclip.set_approx(True)
                c = Collection(r['collection'], r['name'], r['image_path'], r['metadata_path'], fclip, catalog)
                cols.add(c)


def get_cols():
    return cols


def get_collections_name():
    name = []
    for c in cols.get_collections():
        row = []
        row.append(c.get_id())
        row.append(c.get_name())
        name.append(row)
    return name


def get_url(root: str, name: str):
    return str(server_base_path) + "/" + str(root) + "/" + name + ".jpg"


def setParam(image_name: str, collection: str):
    param = []
    char = get_char_image(image_name, collection)
    param.append(char['prod_name'])
    param.append(char['detail_desc'])
    param.append(char['product_type_name'])
    param.append(char['product_group_name'])
    param.append(char['colour_group_name'])
    return param


def get_char_image(name: str, collection: str):
    with open(get_cols().get_collection_from_id(collection).get_metadata_path(), encoding="utf8") as file:
        csv_reader = csv.DictReader(file, delimiter=',')
        for row in csv_reader:
            if str(row['article_id']) == str(name):
                return row
    return None


def retrival_from_text(text: str, col_id: str):
    im = []
    if col_id == "all":
        for c in cols.get_collections():
            row = []
            imgs = []
            r = c.get_fclip().retrieval([text])[0]
            for i in r:
                imgs.append(c.get_catalog()[i])
            row.append(c.get_id())
            row.append(imgs)
            row.append(c.get_name())
            im.append(row)
    else:
        c = cols.get_collection_from_id(col_id)
        row = []
        imgs = []
        r = c.get_fclip().retrieval([text])[0]
        for i in r:
            imgs.append(c.get_catalog()[i])
        row.append(c.get_id())
        row.append(imgs)
        row.append(c.get_name())
        im.append(row)
    return im


def get_label_from_image(url: str, col_id: str):
    im = []
    if col_id == "all":
        for c in cols.get_collections():
            row = []
            imgs = []
            r = c.get_fclip().retrival_img([url])[0]
            for i in r:
                imgs.append(c.get_catalog()[i])
            row.append(c.get_id())
            row.append(imgs)
            row.append(c.get_name())
            im.append(row)
    else:
        c = cols.get_collection_from_id(col_id)
        row = []
        imgs = []
        r = c.get_fclip().retrival_img([url])[0]
        for i in r:
            imgs.append(c.get_catalog()[i])
        row.append(c.get_id())
        row.append(imgs)
        row.append(c.get_name())
        im.append(row)
    return im


class TargetPad:
    def __init__(self, target_ratio: float, size: int, pad_value: Optional[int] = 0):
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
    return Compose([
        TargetPad(target_ratio, dim, pad_value),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
    ])
