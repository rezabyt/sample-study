import re
import os
import argparse

import pickle
import numpy as np

import matplotlib.pyplot as plt

from scipy.special import softmax
from scipy.stats import entropy


def sort_filenames_by_epoch_and_step(directory, data_type):
    # Get all filenames in the directory
    filenames = os.listdir(directory)
    
    # Define a regex pattern to extract epoch and step numbers
    pattern = r"test_{}_epoch_(\d+)_step_(\d+|end)\.pkl".format(data_type)
    
    # Create a list of tuples containing filename, epoch, and step numbers
    file_info = []
    for filename in filenames:
        match = re.match(pattern, filename)
        if match:
            epoch = int(match.group(1))
            step = match.group(2)
            if step == "end":
                step = float('inf')  # Set step to infinity for "end" to ensure it comes last
            else:
                step = int(step)
            file_info.append((filename, epoch, step))
    
    # Sort the list of tuples first by epoch, then by step
    sorted_file_info = sorted(file_info, key=lambda x: (x[1], x[2]))
    
    # Extract filenames from sorted list
    sorted_filenames = [item[0] for item in sorted_file_info]
    
    return sorted_filenames


def read_files_and_gather_values(directory, data_type):
    sorted_filenames = sort_filenames_by_epoch_and_step(directory, data_type)
    
    stats = {'loss': [], 'accuracy': [], 'y_hats': [], 'ys': []}
    for filename in sorted_filenames:
        with open(os.path.join(directory, filename), 'rb') as f:
            data = pickle.load(f)
            for key in stats.keys():
                stats[key].append(data[key])
    
    return {key: np.array(value) for key, value in stats.items()}


def get_stats_over_seeds(args):
    stats = {'loss': [], 'accuracy': [], 'y_hats': [], 'ys': []}

    for seed in args.seeds:
        directory = os.path.join(args.save_path, args.dataset, f'{args.base_exp_name}_seed[{seed}]', 'stats')
        
        current_stats = read_files_and_gather_values(directory, args.data_type)
        for key in stats.keys():
            stats[key].append(current_stats[key])
    
    return {key: np.array(value) for key, value in stats.items()}


def get_avg_stats_over_seeds(stats):
    return {key: np.mean(value, axis=0) for key, value in stats.items()}


def get_point_categories(avg_stats, threshold):
    accuracy, y_hats, ys = avg_stats['accuracy'], avg_stats['y_hats'], avg_stats['ys'][0].astype(int)
    h_hats = softmax(y_hats, axis=-1)
    confidences = h_hats[:, np.arange(len(ys)), ys]
    
    categories = {'non_monotone': [], 'easy': [], 'hard': [], 'monotone': []}
    for i in range(len(ys)):
        confidence = confidences[:, i]
        confidence_drop = np.sum(np.abs(np.diff(confidence)[np.diff(confidence) < 0]))
        
        if confidence_drop > threshold:
            categories['non_monotone'].append(i)
        else:
            distances = {
                'easy': (np.ones_like(confidence) - confidence).sum(),
                'hard': (confidence - np.zeros_like(confidence)).sum(),
                'monotone': np.abs(confidence - accuracy).sum()
            }
            min_category = min(distances, key=distances.get)
            categories[min_category].append(i)
    
    return {key: np.array(value) for key, value in categories.items()}


def plot_entropies(args, category_entropies):
    colors = {'all': 'blue', 'non_monotone': 'red', 'easy': 'green', 'hard': 'orange', 'monotone': 'purple'}
    
    plt.figure()
    for category, entropies in category_entropies.items():
        mean_entropies = np.mean(entropies, axis=0)
        std_entropies = np.std(entropies, axis=0)
        plt.plot(mean_entropies, color=colors[category], label=category)
        plt.fill_between(np.arange(len(mean_entropies)), mean_entropies - std_entropies, mean_entropies + std_entropies, alpha=0.2, color=colors[category])
    
    plt.xlabel('Epoch')
    plt.ylabel('Entropy')
    plt.title('Entropy of Predictions')
    plt.legend()
    
    plt_name = f'results/{args.base_exp_name}_data_type[{args.data_type}]_entropy.png'
    plt.savefig(plt_name, dpi=300)
    plt.close()
        

def get_entropies(y_hats, idx=None):
    y_hats = softmax(y_hats, axis=-1)
    if idx is not None:
        y_hats = y_hats[:, :, idx]
    
    entropies = entropy(y_hats, axis=-1).mean(axis=-1)
    return entropies


def select_subset(stats, start, end=None):
    if end is None:
        end = stats['accuracy'].shape[1]
    return {key: value[:, start:end] for key, value in stats.items()}


def main(args):
    max_steps = None
    # Get average stats over seeds
    stats = get_stats_over_seeds(args)
    stats = select_subset(stats, 0, max_steps)

    # Define categories of points
    avg_stats = get_avg_stats_over_seeds(stats)
    categories = get_point_categories(avg_stats, threshold=3.0)

    category_entropies = {}

    # Get entropies for all points
    entropies = get_entropies(stats['y_hats'])
    category_entropies['all'] = entropies

    # Get entropies for each category
    for category in categories:
        print(f'Category: {category}, Number of points: {len(categories[category])}')
        category_entropies[category] = get_entropies(stats['y_hats'], idx=categories[category])
    
    plot_entropies(args, category_entropies)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Confidence Evaluation')
    parser.add_argument('--base_exp_name', default='', type=str, help='Base experiment name')
    parser.add_argument('--save_path', default='./logs', type=str, help='Path to save checkpoints')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use', choices=['cifar10'])
    parser.add_argument('--data_type', default='clean', type=str, help='Data type to use', choices=['clean', 'adv'])
    parser.add_argument('--seeds', nargs='+', type=int, help='List of seeds')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
