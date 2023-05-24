import os
import pickle
import pandas as pd
from collections import Counter
import torch
import numpy as np

from data_utilis import *
from fashion_clip.fashion_clip import FashionCLIP


fclip = FashionCLIP('fashion-clip')
path = dataset_root / "styles.csv"
articles = pd.read_csv(path, on_bad_lines='skip')
# drop items that have the same description
subset = articles.drop_duplicates("productDisplayName").copy()
# remove items of unkown category
subset = subset[~subset["masterCategory"].isin(["Unknown"])]
# FashionCLIP has a limit of 77 tokens, let's play it safe and drop things with more than 40 tokens
subset = subset[subset["productDisplayName"].apply(lambda x: 4 < len(str(x).split()) < 40)]
# We also drop products types that do not occur very frequently in this subset of data
most_frequent_product_types = [k for k, v in dict(Counter(subset["articleType"].tolist())).items() if v > 10]
subset = subset[subset["articleType"].isin(most_frequent_product_types)]
path_image = dataset_root / 'images'
images = [str(path_image) + "/" + str(k) + ".jpg" for k in subset["id"].tolist()]
for img in images:
    if not os.path.isfile(img):
        images.remove(img)
texts = subset["productDisplayName"].tolist()
images = images[0:10]
texts = texts[0:10]
# we create image embeddings and text embeddings
image_embeddings = fclip.encode_images(images, batch_size=32)
text_embeddings = fclip.encode_text(texts, batch_size=32)
# we normalize the embeddings to unit norm (so that we can use dot product instead of cosine similarity to do comparisons)
image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, ord=2, axis=-1, keepdims=True)
data_path.mkdir(exist_ok=True, parents=True)
torch.save(image_embeddings, data_path / "dataset_index_features.pt")
with open(data_path / f'dataset_index_names.pkl', 'wb+') as f:
    pickle.dump(text_embeddings, f)

# TRY
text_embedding = fclip.encode_text(["trousers"], 32)[0]
id_of_matched_object = np.argmax(text_embedding.dot(image_embeddings.T))
# found_object = subset["productDisplayName"].iloc[id_of_matched_object].tolist()
found_object = subset["productDisplayName"].iloc[id_of_matched_object]
print(found_object)
