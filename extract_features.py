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








    #                                 'caption': row['detail_desc']})
# with open(dataset_root / 'dataset.csv', encoding="utf8") as f:
#     dataset_reader = csv.DictReader(f, delimiter=',')
#     for r in dataset_reader:
#         with open(server_base_path / r['metadata_path'], encoding="utf8") as file:
#             csv_reader = csv.DictReader(file, delimiter=',')
#             catalog = []
#             for row in csv_reader:
#                 catalog.append({'id': row['article_id'], 'image': get_url(r['image_path'], row['article_id']),
#                                 'caption': row['detail_desc']})
#             dataset = FCLIPDataset(name="mydataset", image_source_path=r['image_path'], image_source_type='local',
#                                    catalog=catalog)
#             fclip = FashionCLIP('fashion-clip', dataset, approx=False)
#             p = server_base_path / r['fclip_path']
#             with open(p, 'wb+') as f:
#                 pickle.dump(fclip, f)
