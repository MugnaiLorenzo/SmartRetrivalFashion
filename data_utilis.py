import csv
import pickle
import torchvision.transforms.functional as F
import PIL.Image
import chromadb
import json
import random

from fashion_clip.fashion_clip import FashionCLIP
from pathlib import Path
from typing import Optional
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from collection import Collections, Collection

server_base_path = Path(__file__).absolute().parent.absolute()
dataset_root = Path(__file__).absolute().parent.absolute() / 'dataset'
image_root = dataset_root / 'Images'
data_path = Path(__file__).absolute().parent.absolute() / 'data'


def load():
    global chroma_client
    global fclip
    global n_collection
    global loaded
    loaded = True
    n_collection = 0
    chroma_client = chromadb.Client()
    fclip = FashionCLIP('fashion-clip')
    f = open(dataset_root / 'dataset.json')
    data = json.load(f)
    for i in data:
        n_collection = n_collection + 1
        collection = chroma_client.create_collection(name=i['name'])
        s = str(server_base_path) + "/" + i['metadata_path']
        f = open(s, encoding="utf8")
        d = json.load(f)
        document = []
        ids = []
        metadatas = []
        for j in d:
            document.append(j['detail_desc'])
            ids.append(str(j['article_id']))
            metadatas.append(j)
        p = server_base_path / i['fclip_path']
        with open(p, 'rb') as c:
            image_embeddings = pickle.load(c)
        collection.add(
            embeddings=image_embeddings.tolist(),
            documents=document,
            metadatas=metadatas,
            ids=ids
        )


def get_n_collection():
    return len(chroma_client.list_collections())


def get_collection_from_index(i):
    n = 1
    for col in chroma_client.list_collections():
        if n == i:
            return col
        n = n + 1


def get_collections_name():
    col_name = []
    i = 0
    for col in chroma_client.list_collections():
        i = i + 1
        row = []
        row.append(i)
        row.append(col.name)
        col_name.append(row)
    return col_name


def get_random_images(n_range):
    images = []
    for i in range(0, n_range):
        row = []
        random_c = random.sample(range(get_n_collection()), 1)
        random_c = random_c[0] + 1
        col = get_collection_from_index(random_c)
        n_col = col.count()
        r = random.sample(range(n_col), 1)
        row.append(random_c)
        row.append(col.peek(limit=n_col)['metadatas'][r[0]]['article_id'])
        images.append(row)
    return images


def is_load():
    try:
        if loaded is None:
            return False
        else:
            return True
    except:
        return False


def get_url(root: str, name: str):
    return str(server_base_path) + "/" + str(root) + "/" + name + ".jpg"


def setParam(image_name: str, collection: int):
    col = get_collection_from_index(int(collection))
    param = []
    char = col.get(where={"article_id": image_name})['metadatas'][0]
    param.append(char['prod_name'])  # C
    param.append(char['detail_desc'])
    param.append(char['product_type_name'])  # E
    param.append(char['product_group_name'])  # F
    param.append(char['colour_group_name'])
    return param


def get_collection_for_modify():
    cs = []
    i = 0
    for col in chroma_client.list_collections():
        i = i + 1
        row = []
        row.append(col.name)
        row.append(i)
        n_col = col.count()
        r = random.sample(range(n_col), 1)
        row.append(col.peek(limit=n_col)['metadatas'][r[0]]['article_id'])
        cs.append(row)
    return cs

def get_image_from_collection(id: int):
    col = get_collection_from_index(id)
    n_col = col.count()
    peek = col.peek(limit=n_col)['metadatas']
    return peek, n_col, col.name


def retrival_from_text(text: str, col_id: str):
    im = []
    if col_id == "all":
        index = 0
        for collection in chroma_client.list_collections():
            index = index + 1
            t = []
            t.append(text)
            text_vector = fclip.encode_text(t, batch_size=8)
            r = collection.query(query_embeddings=text_vector.tolist(), n_results=10)
            row = []
            row.append(index)
            row.append(r['ids'][0])
            row.append(collection.name)
            im.append(row)
    else:
        collection = get_collection_from_index(int(col_id))
        t = []
        t.append(text)
        text_vector = fclip.encode_text(t, batch_size=8)
        r = collection.query(query_embeddings=text_vector.tolist(), n_results=10)
        row = []
        row.append(col_id)
        row.append(r['ids'][0])
        row.append(collection.name)
        im.append(row)
    return im


def get_label_from_image(url: str, col_id: str):
    im = []
    if col_id == "all":
        index = 0
        for collection in chroma_client.list_collections():
            index = index + 1
            t = []
            t.append(url)
            image_vector = fclip.encode_images(t, batch_size=8)
            r = collection.query(query_embeddings=image_vector.tolist(), n_results=10)
            row = []
            row.append(index)
            row.append(r['ids'][0])
            row.append(collection.name)
            im.append(row)
    else:
        collection = get_collection_from_index(int(col_id))
        t = []
        t.append(url)
        image_vector = fclip.encode_images(t, batch_size=8)
        r = collection.query(query_embeddings=image_vector.tolist(), n_results=10)
        row = []
        row.append(col_id)
        row.append(r['ids'][0])
        row.append(collection.name)
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
