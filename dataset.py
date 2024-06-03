import os
import re
import torch
import torchvision as torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from semisup import get_semisup_dataloaders
from semisup import SemiSupervisedDataset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_normalization(dataset):
    if dataset in ('cifar10', 'cifar10s'):
        return CIFAR10_MEAN, CIFAR10_STD
    elif dataset in ('cifar100', 'cifar100s'):
        return CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(f'Dataset not recognized ({dataset})')


def get_data_loaders(args, data_path, norm=False, train_shuffle=True):
    if args.dataset == 'cifar10':
        train_loader = get_cifar10_train_loader(args=args, data_path=data_path, norm=norm, shuffle=train_shuffle)
        test_loader = get_cifar10_test_loader(args=args, data_path=data_path, norm=norm)
    elif args.dataset == 'cifar10s':
        train_loader, test_loader = get_cifar10s_loaders(args=args, data_path=data_path, norm=norm)
    elif args.dataset == 'cifar100':
        train_loader = get_cifar100_train_loader(args=args, data_path=data_path, norm=norm, shuffle=train_shuffle)
        test_loader = get_cifar100_test_loader(args=args, data_path=data_path, norm=norm)
    elif args.dataset == 'cifar100s':
        train_loader, test_loader = get_cifar100s_loaders(args=args, data_path=data_path, norm=norm)
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


def get_cifar10s_loaders(args, data_path, norm):
    train_dataset, test_dataset = load_cifar10s(data_path=data_path, aux_data_filename=args.aux_data_filename, aux_take_amount=args.aux_take_amount, norm=norm)
    
    train_loader, test_loader = get_semisup_dataloaders(train_dataset=train_dataset, test_dataset=test_dataset,
                                                        train_batch_size=args.batch_size, test_batch_size=args.batch_size,
                                                        num_workers=8, unsup_fraction=args.unsup_fraction)
    train_loader.name = "train"
    test_loader.name = "test"
    return train_loader, test_loader


def load_cifar10s(data_path, aux_data_filename='', aux_take_amount=None, norm=False):
    """
    Load CIFAR10 dataset with auxiliary data.

    Args:
        data_path (str): Path to the data.
        aux_data_filename (str): Path to the auxiliary data.
        aux_take_amount (int): Amount of auxiliary data to take. None means all.
        norm (bool): Normalize the data.

    Returns:
        train_dataset (SemiSupervisedCIFAR10): Semi-supervised CIFAR10 training dataset.
        test_dataset (SemiSupervisedCIFAR10): Semi-supervised CIFAR10 test dataset.
    """
    
    # Change the data path to the original CIFAR10 path for the base dataset
    base_data_path = re.sub('cifar10s', 'cifar10', data_path)
    
    test_ts = [transforms.ToTensor()]
    if norm:
        test_ts.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    test_transforms = transforms.Compose(test_ts)
    
    trian_ts = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    if norm:
        trian_ts.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    train_transforms = transforms.Compose(trian_ts)
    
    train_dataset = SemiSupervisedCIFAR10(base_dataset='cifar10', root=base_data_path, train=True, download=True, 
                                          transform=train_transforms, aux_data_filename=aux_data_filename, 
                                          add_aux_labels=True, aux_take_amount=aux_take_amount)
    test_dataset = SemiSupervisedCIFAR10(base_dataset='cifar10', root=base_data_path, train=False, download=True, 
                                         transform=test_transforms)
    return train_dataset, test_dataset


class SemiSupervisedCIFAR10(SemiSupervisedDataset):
    """
    A dataset with auxiliary pseudo-labeled data for CIFAR10.
    """
    def load_base_dataset(self, train=False, **kwargs):
        assert self.base_dataset == 'cifar10', 'Only semi-supervised cifar10 is supported. Please use correct dataset!'
        self.dataset = torchvision.datasets.CIFAR10(train=train, **kwargs)
        self.dataset_size = len(self.dataset)


def get_cifar100s_loaders(args, data_path, norm):
    train_dataset, test_dataset = load_cifar100s(data_path=data_path, aux_data_filename=args.aux_data_filename, aux_take_amount=args.aux_take_amount, norm=norm)
    
    train_loader, test_loader = get_semisup_dataloaders(train_dataset=train_dataset, test_dataset=test_dataset,
                                                        train_batch_size=args.batch_size, test_batch_size=args.batch_size,
                                                        num_workers=8, unsup_fraction=args.unsup_fraction)
    train_loader.name = "train"
    test_loader.name = "test"
    return train_loader, test_loader


def load_cifar100s(data_path, aux_data_filename='', aux_take_amount=None, norm=False):
    """
    Load CIFAR100 dataset with auxiliary data.

    Args:
        data_path (str): Path to the data.
        aux_data_filename (str): Path to the auxiliary data.
        aux_take_amount (int): Amount of auxiliary data to take. None means all.
        norm (bool): Normalize the data.

    Returns:
        train_dataset (SemiSupervisedCIFAR100): Semi-supervised CIFAR100 training dataset.
        test_dataset (SemiSupervisedCIFAR100): Semi-supervised CIFAR100 test dataset.
    """
    
    # Change the data path to the original CIFAR100 path for the base dataset
    base_data_path = re.sub('cifar100s', 'cifar100', data_path)
    
    test_ts = [transforms.ToTensor()]
    if norm:
        test_ts.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD))
    test_transforms = transforms.Compose(test_ts)
    
    trian_ts = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    if norm:
        trian_ts.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD))
    train_transforms = transforms.Compose(trian_ts)
    
    train_dataset = SemiSupervisedCIFAR100(base_dataset='cifar100', root=base_data_path, train=True, download=True, 
                                           transform=train_transforms, aux_data_filename=aux_data_filename, 
                                           add_aux_labels=True, aux_take_amount=aux_take_amount)
    test_dataset = SemiSupervisedCIFAR100(base_dataset='cifar100', root=base_data_path, train=False, download=True, 
                                          transform=test_transforms)
    return train_dataset, test_dataset


class SemiSupervisedCIFAR100(SemiSupervisedDataset):
    """
    A dataset with auxiliary pseudo-labeled data for CIFAR100.
    """
    def load_base_dataset(self, train=False, **kwargs):
        assert self.base_dataset == 'cifar100', 'Only semi-supervised cifar100 is supported. Please use correct dataset!'
        self.dataset = torchvision.datasets.CIFAR100(train=train, **kwargs)
        self.dataset_size = len(self.dataset)
