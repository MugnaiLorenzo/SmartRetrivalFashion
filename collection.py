import random


class Collection:
    def __init__(self, collection, name, image_path, metadata_path, fclip, catalog):
        self.id = collection
        self.name = name
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.fclip = fclip
        self.catalog = catalog

    def get_id(self):
        return self.id

    def get_name(self):
        return self.name

    def get_image_path(self):
        return self.image_path

    def get_metadata_path(self):
        return self.metadata_path

    def get_fclip(self):
        return self.fclip

    def get_catalog(self):
        return self.catalog

    def get_random_image(self):
        r = random.sample(range(len(self.catalog)), k=1)
        return self.catalog[r[0]]


class Collections:
    def __init__(self):
        self.collections = []

    def add(self, collection):
        self.collections.append(collection)

    def get_collection_from_id(self, id: str):
        for c in self.collections:
            if str(c.get_id()) == str(id):
                return c
        return None

    def get_collections(self):
        return self.collections

    def get_len(self):
        return len(self.collections)
