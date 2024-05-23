import re
import os
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm

from utils import get_model

import matplotlib.pyplot as plt



# Define the dataloader for CIFAR10-neg
class CIFAR10_neg(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
    

def get_cifar10_neg_loader(data_path='data/CIFAR10_neg.npz'):
    CIFAR_10_neg = np.load(data_path)
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_neg = CIFAR10_neg(CIFAR_10_neg['data'], CIFAR_10_neg['labels'], transform)
    cifar10_neg_loader = DataLoader(cifar10_neg, batch_size=128, shuffle=False)
    return cifar10_neg_loader  


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


def evaluate_checkpoints_on_cifar10_neg(args, loader, device):
    seed_accuracy = []
    for seed in tqdm(args.seeds):
        directory = os.path.join(args.save_path, args.dataset, f'{args.base_exp_name}_seed[{seed}]', 'checkpoints')
        sorted_checkpoints = sort_checkpoints_by_epoch_and_step(directory)
        accuracy = []
        for checkpoint in sorted_checkpoints:
            model = load_model(args, directory, checkpoint, device)
            acc = get_accuracy(model, loader, device)
            accuracy.append(acc)
        seed_accuracy.append(accuracy)
    
    return np.array(seed_accuracy)


def plot_cifar10_neg_accuracy(accuracies):    
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})

    colos = {'nt': 'blue', 'at': 'red'}

    for key, accuracy in accuracies.items():
        mean_accuracy = np.mean(accuracy, axis=0)
        std_accuracy = np.std(accuracy, axis=0)

        epochs = np.arange(0, len(mean_accuracy))

        plt.plot(epochs, mean_accuracy, label=key.upper(), color=colos[key], alpha=0.8)
        plt.fill_between(epochs, mean_accuracy - std_accuracy, mean_accuracy + std_accuracy, color=colos[key], alpha=0.2)

    plt.xlabel('Epoch', labelpad=10)
    plt.ylabel('Accuracy', labelpad=20)
    plt.title('CIFAR10-neg Accuracy', pad=10)
    plt.ylim(0.1, 0.7)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt_name = f'results/cifar10_neg/cifar10_neg_accuracy_16.png'
    plt.savefig(plt_name, dpi=300)
    plt.close()


def visualization():
    # Plot CIFAR10-neg accuracy
    nt_accuracy_path = 'trainer[nt]_model[resnet18]_epochs[120]_bs[128]_dataset[cifar10]_opt[sgd]_lr[0.1]_lr_scheduler[multi]'
    at_accuracy_path = 'trainer[at]_model[resnet18]_epochs[120]_bs[128]_dataset[cifar10]_opt[sgd]_lr[0.1]_lr_scheduler[multi]_attack[pgd]'

    max_epoch = 100
    nt_accuracy = np.load(f'results/cifar10_neg/{nt_accuracy_path}_cifar10_neg_accuracy.npy')[:, :max_epoch]
    at_accuracy = np.load(f'results/cifar10_neg/{at_accuracy_path}_cifar10_neg_accuracy.npy')[:, :max_epoch]

    accuracies = {
        'nt': nt_accuracy,
        'at': at_accuracy
    }
    plot_cifar10_neg_accuracy(accuracies)


def main(args):
    # Use GPUs if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    # Load CIFAR10-neg dataset
    cifar10_neg_loader = get_cifar10_neg_loader()
    
    # Evaluate checkpoints on CIFAR10-neg
    accuracy = evaluate_checkpoints_on_cifar10_neg(args, cifar10_neg_loader, device)
    np.save(f'results/cifar10_neg/{args.base_exp_name}_cifar10_neg_accuracy.npy', accuracy)

    # Visualization
    # visualization()


def parse_arguments():
    parser = argparse.ArgumentParser(description='CIFAR10_NEG Evaluation')
    parser.add_argument('--base_exp_name', default='', type=str, help='Base experiment name')
    parser.add_argument('--save_path', default='./logs', type=str, help='Path to save checkpoints')
    parser.add_argument('--model', default='resnet18', type=str, help='Choose the model.', choices=['resnet18'])
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use', choices=['cifar10', 'cifar10s'])
    parser.add_argument('--seeds', nargs='+', type=int, help='List of seeds')

    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
