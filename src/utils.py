from models.resnet import resnet18, resnet34, resnet50, resnet152, resnet101
from models.efficientnet import EfficientNetFc


import torchvision.transforms as transforms
from data.tranforms import TwoCropsTransform
import data
from data.cifar100 import get_cifar100
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
import torch



def load_data(args):
    batch_size_dict = {"train": args.batch_size, "unlabeled_train": args.batch_size, "test": 100}

    if 'cifar100' in args.root:
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, args.root)
        labeled_trainloader = DataLoader(
            labeled_dataset,
            sampler=RandomSampler(labeled_dataset),
            batch_size=batch_size_dict["train"],
            num_workers=4,
            drop_last=True)

        unlabeled_trainloader = DataLoader(
            unlabeled_dataset,
            sampler=RandomSampler(unlabeled_dataset),
            batch_size=batch_size_dict["unlabeled_train"],
            num_workers=4,
            drop_last=True)

        test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=batch_size_dict["test"],
            num_workers=4)

        dataset_loaders = {"train": labeled_trainloader,
                        "unlabeled_train": unlabeled_trainloader,
                        "test": test_loader}
    else:
        data_transforms = {}
        data_transforms['train'] = TwoCropsTransform()
        data_transforms['test'] = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        dataset = data.__dict__[os.path.basename(args.root)]

        datasets = {"train": dataset(root=args.root, split='train', label_ratio=args.label_ratio,
                                download=True, transform=data_transforms["train"]),
                 "unlabeled_train": dataset(root=args.root, split='unlabeled_train', label_ratio=args.label_ratio,
                                download=True, transform=data_transforms["train"]),
                 "test": dataset(root=args.root, split='test', label_ratio=100,
                                download=True, transform=data_transforms["test"])}

        dataset_loaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size_dict[x],
                                                       shuffle=('train' in x), num_workers=4)
                        for x in ['train', 'unlabeled_train', 'test']}

    return dataset_loaders



def load_network(backbone):
    if 'resnet' in backbone:
        if backbone == 'resnet18':
            network = resnet18
            feature_dim = 512
        elif backbone == 'resnet34':
            network = resnet34
            feature_dim = 512
        elif backbone == 'resnet50':
            network = resnet50
            feature_dim = 2048
        elif backbone == 'resnet101':
            network = resnet101
            feature_dim = 2048
        elif backbone == 'resnet152':
            network = resnet152
            feature_dim = 2048
    elif 'efficientnet' in backbone:
        network = EfficientNetFc
        print(backbone)
        if backbone == 'efficientnet-b0':
            feature_dim = 1280
        elif backbone == 'efficientnet-b1':
            feature_dim = 1280
        elif backbone == 'efficientnet-b2':
            feature_dim = 1408
        elif backbone == 'efficientnet-b3':
            feature_dim = 1536
        elif backbone == 'efficientnet-b4':
            feature_dim = 1792
        elif backbone == 'efficientnet-b5':
            feature_dim = 2048
        elif backbone == 'efficientnet-b6':
            feature_dim = 2304
    else:
        network = resnet50
        feature_dim = 2048

    return network, feature_dim