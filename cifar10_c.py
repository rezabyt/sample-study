
import os
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from tqdm import tqdm

from models.resnet import ResNet18


class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root_dir, corruption_type, transform=None):
        super(CIFAR10C, self).__init__(
            root_dir, transform=transform)
        data_path = os.path.join(root_dir, corruption_type + '.npy')
        target_path = os.path.join(root_dir, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)


def get_cifar10c_loader(args):
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])
    
    dataset = CIFAR10C(args.data_path, args.cname, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    return dataset, loader


def load_model():
    model = ResNet18()
    return model


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(loader):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def main(args):
    _, loader = get_cifar10c_loader(args)

    model = load_model()

    acc = evaluate(model, loader)
    print(f'Accuracy on {args.cname}: {acc}')
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='CIFAR10-C Evaluation')
    parser.add_argument('--base_exp_name', default='', type=str, help='Base experiment name')
    parser.add_argument('--save_path', default='./logs', type=str, help='Path to save checkpoints')
    parser.add_argument('--model', default='resnet18', type=str, help='Choose the model.', choices=['resnet18'])
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use', choices=['cifar10', 'cifar10s'])
    parser.add_argument('--seeds', nargs='+', type=int, help='List of seeds')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)