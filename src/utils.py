from models.resnet import resnet18, resnet34, resnet50, resnet152, resnet101
from models.efficientnet import EfficientNetFc

from data.tranforms import TransformTrain
from data.tranforms import TransformTest
import data
from data.cifar100 import get_cifar100
from torch.utils.data import DataLoader, RandomSampler
import os

imagenet_mean=(0.485, 0.456, 0.406)
imagenet_std=(0.229, 0.224, 0.225)

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

        ## We didn't apply tencrop test since other SSL baselines neither
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size_dict["test"],
            shuffle=False,
            num_workers=4)

        dataset_loaders = {"train": labeled_trainloader,
                           "unlabeled_train": unlabeled_trainloader,
                           "test": test_loader}

    else:
        transform_train = TransformTrain()
        transform_test = TransformTest(mean=imagenet_mean, std=imagenet_std)
        dataset = data.__dict__[os.path.basename(args.root)]

        datasets = {"train": dataset(root=args.root, split='train', label_ratio=args.label_ratio, download=True, transform=transform_train),
                    "unlabeled_train": dataset(root=args.root, split='unlabeled_train', label_ratio=args.label_ratio, download=True, transform=transform_train)}
        test_dataset = {
            'test' + str(i): dataset(root=args.root, split='test', label_ratio=100, download=True, transform=transform_test["test" + str(i)]) for i in range(10)
        }
        datasets.update(test_dataset)

        dataset_loaders = {x: DataLoader(datasets[x], batch_size=batch_size_dict[x], shuffle=True, num_workers=4)
                           for x in ['train', 'unlabeled_train']}
        dataset_loaders.update({'test' + str(i): DataLoader(datasets["test" + str(i)], batch_size=4, shuffle=False, num_workers=4)
                                for i in range(10)})

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