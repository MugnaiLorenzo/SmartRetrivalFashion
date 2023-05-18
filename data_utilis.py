from pathlib import Path

server_base_path = Path(__file__).absolute().parent.absolute()
dataset_root = Path(__file__).absolute().parent.absolute() / 'dataset'


# check is an image
def is_image_file(filename: str) -> bool:
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
