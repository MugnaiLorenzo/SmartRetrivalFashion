import csv
from pathlib import Path
from typing import Optional
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms.functional as F
import PIL.Image

server_base_path = Path(__file__).absolute().parent.absolute()
dataset_root = Path(__file__).absolute().parent.absolute() / 'data_set' / 'fashion-dataset'
image_root = dataset_root / 'images'
image_id = []


# check is an image
def is_image_file(filename: str) -> bool:
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


# conversion image to rgb
def _convert_image_to_rgb(image):
    return image.convert("RGB")


# read cvs file
def read_cvs():
    with open(dataset_root / 'styles.csv') as file:
        csv_reader = csv.DictReader(file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                image_id.append(row['id'])
            line_count += 1


def get_char_image(name: str):
    with open(dataset_root / 'styles.csv') as file:
        csv_reader = csv.DictReader(file, delimiter=',')
        for row in csv_reader:
            if row['id'] == name:
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
