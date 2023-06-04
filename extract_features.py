import pickle
import csv

from fashion_clip.fashion_clip import FashionCLIP, FCLIPDataset
from data_utilis import dataset_root, server_base_path, get_url

with open(dataset_root / 'dataset.csv', encoding="utf8") as f:
    dataset_reader = csv.DictReader(f, delimiter=',')
    for r in dataset_reader:
        with open(server_base_path / r['metadata_path'], encoding="utf8") as file:
            csv_reader = csv.DictReader(file, delimiter=',')
            catalog = []
            for row in csv_reader:
                catalog.append({'id': row['article_id'], 'image': get_url(r['image_path'], row['article_id']),
                                'caption': row['detail_desc']})
            dataset = FCLIPDataset(name="mydataset", image_source_path=r['image_path'], image_source_type='local',
                                   catalog=catalog)
            fclip = FashionCLIP('fashion-clip', dataset, approx=False)
            p = server_base_path / r['fclip_path']
            with open(p, 'wb+') as f:
                pickle.dump(fclip, f)
