import os
from typing import Optional
from .imagelist import ImageList
from ._util import *


class Aircraft(ImageList):
    """`FVGC-Aircraft <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        label_ratio (int): The sampling rates to sample random ``training`` images for each category.
            Choices include 100, 50, 30, 15. Default: 100.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            train/
            test/
            image_list/
                train_100.txt
                train_50.txt
                train_30.txt
                train_15.txt
                test.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/04356d49d0054092b07e/?dl=1"),
        ("train", "train.tgz", "https://cloud.tsinghua.edu.cn/f/9fed22eba03046d69012/?dl=1"),
        ("test", "test.tgz", "https://cloud.tsinghua.edu.cn/f/8d5e4c1b031a4a608c68/?dl=1"),
    ]
    image_list = {
        "train": "image_list/train_100.txt",
        "train100": "image_list/train_100.txt",
        "train50": "image_list/train_50.txt",
        "train30": "image_list/train_30.txt",
        "train15": "image_list/train_15.txt",
        "test": "image_list/test.txt",
        "test100": "image_list/test.txt",
    }
    CLASSES = ['707-320', '727-200', '737-200', '737-300', '737-400', '737-500', '737-600', '737-700', '737-800', '737-900', '747-100',
               '747-200', '747-300', '747-400', '757-200', '757-300', '767-200', '767-300', '767-400', '777-200', '777-300', 'A300B4',
               'A310', 'A318', 'A319', 'A320', 'A321', 'A330-200', 'A330-300', 'A340-200', 'A340-300', 'A340-500', 'A340-600', 'A380',
               'ATR-42', 'ATR-72', 'An-12', 'BAE 146-200', 'BAE 146-300', 'BAE-125', 'Beechcraft 1900', 'Boeing 717', 'C-130', 'C-47',
               'CRJ-200', 'CRJ-700', 'CRJ-900', 'Cessna 172', 'Cessna 208', 'Cessna 525', 'Cessna 560', 'Challenger 600', 'DC-10',
               'DC-3', 'DC-6', 'DC-8', 'DC-9-30', 'DH-82', 'DHC-1', 'DHC-6', 'DHC-8-100', 'DHC-8-300', 'DR-400', 'Dornier 328', 'E-170',
               'E-190', 'E-195', 'EMB-120', 'ERJ 135', 'ERJ 145', 'Embraer Legacy 600', 'Eurofighter Typhoon', 'F-16A-B', 'F-A-18',
               'Falcon 2000', 'Falcon 900', 'Fokker 100', 'Fokker 50', 'Fokker 70', 'Global Express', 'Gulfstream IV', 'Gulfstream V',
               'Hawk T1', 'Il-76', 'L-1011', 'MD-11', 'MD-80', 'MD-87', 'MD-90', 'Metroliner', 'Model B200', 'PA-28', 'SR-20',
               'Saab 2000', 'Saab 340', 'Spitfire', 'Tornado', 'Tu-134', 'Tu-154', 'Yak-42']

    def __init__(self, root: str, split: str, label_ratio: Optional[int]=100, download: Optional[bool] = False, **kwargs):

        if split == 'train':
            list_name = 'train' + str(label_ratio)
            assert list_name in self.image_list
            data_list_file = os.path.join(root, self.image_list[list_name])
        elif split == 'unlabeled_train':
            data_list_file = os.path.join(root, "image_list/unlabeled_" + str(label_ratio) + ".txt")
            # if not os.path.exists(data_list_file):
            train_list_name = 'train' + str(label_ratio)
            full_list_name = 'train'
            assert train_list_name in self.image_list
            assert full_list_name in self.image_list
            train_list_file = os.path.join(root, self.image_list[train_list_name])
            full_list_file = os.path.join(root, self.image_list[full_list_name])
            train_list = read_list_from_file(train_list_file)
            full_list = read_list_from_file(full_list_file)
            unlabel_list = list(set(full_list) - set(train_list))
            save_list_to_file(data_list_file, unlabel_list)
        else:
            data_list_file = os.path.join(root, self.image_list['test'])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(Aircraft, self).__init__(root, Aircraft.CLASSES, data_list_file=data_list_file, **kwargs)