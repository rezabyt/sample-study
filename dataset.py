import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def get_data_loaders(args, data_path, norm=False):
    if args.dataset == 'cifar10':
        train_loader = get_cifar10_train_loader(args=args, data_path=data_path, norm=norm)
        test_loader = get_cifar10_test_loader(args=args, data_path=data_path, norm=norm)
    else:
        raise ValueError(f'Dataset not recognized ({args.dataset})')
    return train_loader, test_loader


def get_cifar10_train_loader(args, data_path, norm):
    ts = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    if norm:
        ts.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))
    train_transforms = transforms.Compose(ts)

    dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    loader.name = "train_cifar10"

    return loader


def get_cifar10_test_loader(args, data_path, norm=False):
    ts = [transforms.ToTensor()]
    if norm:
        ts.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))
    test_transforms = transforms.Compose(ts)
    dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=test_transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    loader.name = "test_cifar10"
    return loader
