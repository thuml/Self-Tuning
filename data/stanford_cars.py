import os
from typing import Optional
from .imagelist import ImageList
from ._util import *


class StanfordCars(ImageList):
    """`The Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        label_ratio (int): The label rates to randomly sample ``training`` images for each category.
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
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/d95c188cc49c404aba70/?dl=1"),
        ("train", "train.tgz", "https://cloud.tsinghua.edu.cn/f/d5ab63c391a949509db0/?dl=1"),
        ("test", "test.tgz", "https://cloud.tsinghua.edu.cn/f/04e6fd5222a84d0a8ff5/?dl=1"),
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
    CLASSES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
               '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
               '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
               '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80',
               '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100',
               '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120',
               '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
               '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160',
               '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180',
               '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196']

    def __init__(self, root: str, split: str, label_ratio: Optional[int] =100, download: Optional[bool] = False, **kwargs):

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

        super(StanfordCars, self).__init__(root, StanfordCars.CLASSES, data_list_file=data_list_file, **kwargs)
