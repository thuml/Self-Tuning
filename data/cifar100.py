import math
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from data.tranforms import TransformTrainCifar
from data.tranforms import ResizeImage

crop_size = 224
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)


def get_cifar100(args, root):
    transform_val = transforms.Compose([
        ResizeImage(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])
    base_dataset = datasets.CIFAR100(root, train=True, download=True)
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)
    print("train_labeled_idxs: ", len(train_labeled_idxs))
    print("train_unlabeled_idxs: ", len(train_unlabeled_idxs))

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=TransformTrainCifar(mean=cifar100_mean, std=cifar100_std))

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformTrainCifar(mean=cifar100_mean, std=cifar100_std))
    test_dataset = datasets.CIFAR100(root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.class_num
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.class_num):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_label or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.test_interval / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx



class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
