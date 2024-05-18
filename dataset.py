import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_normalization(dataset):
    if dataset == 'cifar10':
        return CIFAR10_MEAN, CIFAR10_STD
    elif dataset == 'cifar100':
        return CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(f'Dataset not recognized ({dataset})')


def get_data_loaders(args, data_path, norm=False, train_shuffle=True):
    if args.dataset == 'cifar10':
        train_loader = get_cifar10_train_loader(args=args, data_path=data_path, norm=norm, shuffle=train_shuffle)
        test_loader = get_cifar10_test_loader(args=args, data_path=data_path, norm=norm)
    elif args.dataset == 'cifar100':
        train_loader = get_cifar100_train_loader(args=args, data_path=data_path, norm=norm, shuffle=train_shuffle)
        test_loader = get_cifar100_test_loader(args=args, data_path=data_path, norm=norm)
    else:
        raise ValueError(f'Dataset not recognized ({args.dataset})')
    return train_loader, test_loader


def get_cifar10_train_loader(args, data_path, norm, shuffle=True):
    ts = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    if norm:
        ts.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    train_transforms = transforms.Compose(ts)

    dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=8)
    loader.name = "train"

    return loader


def get_cifar10_test_loader(args, data_path, norm=False):
    ts = [transforms.ToTensor()]
    if norm:
        ts.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    test_transforms = transforms.Compose(ts)
    dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=test_transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    loader.name = "test"
    return loader


def get_cifar100_train_loader(args, data_path, norm, shuffle=True):
    ts = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    if norm:
        ts.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD))
    train_transforms = transforms.Compose(ts)

    dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=8)
    loader.name = "train"

    return loader


def get_cifar100_test_loader(args, data_path, norm=False):
    ts = [transforms.ToTensor()]
    if norm:
        ts.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD))
    test_transforms = transforms.Compose(ts)
    dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=test_transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    loader.name = "test"
    return loader
