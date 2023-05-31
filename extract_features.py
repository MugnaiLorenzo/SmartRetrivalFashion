import os
import pickle
import pandas as pd
from collections import Counter
import torch
import numpy as np
import csv

from fashion_clip.fashion_clip import FashionCLIP, FCLIPDataset
from data_utilis import dataset_root, image_root, get_url, data_path

with open(dataset_root / 'articles.csv', encoding="utf8") as file:
    csv_reader = csv.DictReader(file, delimiter=',')
    line_count = 0
    catalog = []
    for row in csv_reader:
        if line_count != 0:
            catalog.append(
                {'id': row['article_id'], 'image': get_url(row['article_id']), 'caption': row['detail_desc']})
        line_count += 1
dataset = FCLIPDataset(name="mydataset", image_source_path=str(image_root), image_source_type='local', catalog=catalog)
fclip = FashionCLIP('fashion-clip', dataset, approx=False)
with open(data_path / f'f_clip.pkl', 'wb+') as f:
    pickle.dump(fclip, f)

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
images = []
for k in subset["article_id"].tolist():
    if os.path.isfile(str(path_image) + "/" + str(k) + ".jpg"):
        images.append(str(path_image) + "/" + str(k) + ".jpg")
texts = subset["detail_desc"].tolist()
# we create image embeddings and text embeddings
image_embeddings = fclip.encode_images(images, batch_size=32)
text_embeddings = fclip.encode_text(texts, batch_size=32)
# we normalize the embeddings to unit norm (so that we can use dot product instead of cosine similarity to do
# comparisons)
image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, ord=2, axis=-1, keepdims=True)
data_path.mkdir(exist_ok=True, parents=True)
torch.save(image_embeddings, data_path / "dataset_index_features.pt")
with open(data_path / f'dataset_index_names.pkl', 'wb+') as f:
    pickle.dump(text_embeddings, f)
