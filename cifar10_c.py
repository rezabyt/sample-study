import re
import os
import time
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from utils import get_model


class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root_dir, cname, transform=None):
        super(CIFAR10C, self).__init__(
            root_dir, transform=transform)
        root_dir = os.path.join(root_dir, 'CIFAR-10-C')

        data_path = os.path.join(root_dir, f'{cname}.npy')
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


def get_cifar10c_loader(args, cname): 
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR10C(args.data_path, cname, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    return dataset, loader


def load_model(args, directory, checkpoint, device):
    checkpoint_path = os.path.join(directory, checkpoint)
    model = get_model(dataset=args.dataset, model=args.model)
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model = model.to(device)
    return model.eval()


def get_accuracy(model, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def sort_checkpoints_by_epoch_and_step(directory):
    # Get all checkpoints in the directory
    checkpoints = os.listdir(directory)
    
    # Define a regex pattern to extract epoch and step numbers
    pattern = r"epoch_(\d+)_step_(\d+|end)\.pt"
    
    # Create a list of tuples containing checkpoint, epoch, and step numbers
    checkpoint_info = []
    for checkpoint in checkpoints:
        match = re.match(pattern, checkpoint)
        if match:
            epoch = int(match.group(1))
            step = match.group(2)
            if step == "end":
                step = float('inf')  # Set step to infinity for "end" to ensure it comes last
            else:
                step = int(step)
            checkpoint_info.append((checkpoint, epoch, step))
    
    # Sort the list of tuples first by epoch, then by step
    sorted_checkpoint_info = sorted(checkpoint_info, key=lambda x: (x[1], x[2]))
    
    # Extract checkpoints from sorted list
    sorted_checkpoints = [item[0] for item in sorted_checkpoint_info]
    
    return sorted_checkpoints


def evaluate_checkpoints_on_cifar10_c(args, loader, device):
    seed_accuracy = []
    for seed in args.seeds:
        directory = os.path.join(args.save_path, args.dataset, f'{args.base_exp_name}_seed[{seed}]', 'checkpoints')
        sorted_checkpoints = [sort_checkpoints_by_epoch_and_step(directory)[-1]]
        accuracy = []
        for checkpoint in sorted_checkpoints:
            model = load_model(args, directory, checkpoint, device)
            acc = get_accuracy(model, loader, device)
            accuracy.append(acc)
        seed_accuracy.append(accuracy)
    
    return np.array(seed_accuracy)


def all_cnames():
    cnames = [
        "brightness",
        "defocus_blur",
        "fog",
        "gaussian_blur",
        "glass_blur",
        "jpeg_compression",
        "motion_blur",
        "saturate",
        "snow",
        "speckle_noise",
        "contrast",
        "elastic_transform",
        "frost",
        "gaussian_noise",
        "impulse_noise",
        "pixelate",
        "shot_noise",
        "spatter",
        "zoom_blur"
    ]
    return cnames


def main(args):
    # Use GPUs if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    start_time = time.time()
    # Evaluate checkpoints on CIFAR10-C
    all_accuracy = []
    for cname in all_cnames():
        _, cifar10c_loader = get_cifar10c_loader(args, cname)
        accuracy = evaluate_checkpoints_on_cifar10_c(args, cifar10c_loader, device)
        all_accuracy.append(accuracy)
        print(f'{cname}: mean: {accuracy.mean():.4f}, std: {accuracy.std():.4f}')
    
    print(f'Total time taken: {time.time() - start_time}')
    all_accuracy = np.array(all_accuracy)   
    np.save(f'results/cifar10_c/{args.base_exp_name}_cifar10_c_accuracy.npy', accuracy)
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='CIFAR10-C Evaluation')
    parser.add_argument('--base_exp_name', default='', type=str, help='Base experiment name')
    parser.add_argument('--data_path', default='./data', type=str, help='Path to data directory')
    parser.add_argument('--save_path', default='./logs', type=str, help='Path to save checkpoints')
    parser.add_argument('--model', default='resnet18', type=str, help='Choose the model.', choices=['resnet18'])
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use', choices=['cifar10', 'cifar10s'])
    parser.add_argument('--seeds', nargs='+', type=int, help='List of seeds')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)