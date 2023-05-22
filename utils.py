import json
import os

import cv2
import numpy as np
import torch


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_uri_to_metadata_dict(NOISYART_IMGS_PATH, NOISYART_JSON_PATH):
    '''
    align the metadata.json classes order to the classes names order
    :param NOISYART_IMGS_PATH: noisyart classes path
    :param NOISYART_JSON_PATH: noisyart path to `metadata.json`
    :return: the aligned metadata
    '''
    classes = sorted(os.listdir(NOISYART_IMGS_PATH))
    classes_dict = {}
    for c in classes:
        classes_dict[c[5:]] = int(c[0:4])

    with open(NOISYART_JSON_PATH, 'r') as f:
        metadata = json.loads(f.read())

    uri_to_metadata = {m['uri'].replace('/', '^'): m for m in metadata}

    # sorted_metadata = [uri_to_metadata[uri] for uri in
    #                    classes_dict]  # It works in python 3.7+ where dictionary iteration order is guaranteed to be in order of insertion.

    return uri_to_metadata
