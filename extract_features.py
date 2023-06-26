import pickle
import csv
import json

from fashion_clip.fashion_clip import FashionCLIP, FCLIPDataset
from data_utilis import dataset_root, server_base_path, get_url

f = open(dataset_root / 'dataset.json')
data = json.load(f)
fclip = FashionCLIP('fashion-clip')
for i in data:
    s = str(server_base_path) + "/" + i['metadata_path']
    f = open(s, encoding="utf8")
    d = json.load(f)
    catalog = []
    images = []
    for j in d:
        path = str(server_base_path) + "/" + str(i['image_path']) + "/" + str(j['article_id']) + ".jpg"
        catalog.append(
            {'id': j['article_id'], 'image': path, 'caption': j['detail_desc']})
        images.append(path)
    images_embedded = fclip.encode_images(images, batch_size=8)
    p = server_base_path / i['fclip_path']
    with open(p, 'wb+') as f:
        pickle.dump(images_embedded, f)
