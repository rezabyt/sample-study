import re
import os
import argparse

import pickle
import numpy as np

import matplotlib.pyplot as plt

from scipy.special import softmax
from scipy.stats import entropy


CATEGORY_COLORS = {'monotone': 'purple', 'easy': 'green', 'hard': 'red', 'non_monotone': 'orange'}
ALL_COLORS = CATEGORY_COLORS.copy()
ALL_COLORS['all'] = 'blue'

CATEGORY_LABELS = {'monotone': 'Monotone', 'easy': 'Easy', 'hard': 'Hard', 'non_monotone': 'Non-Monotone'}
ALL_LABELS = CATEGORY_LABELS.copy()
ALL_LABELS['all'] = 'All'

def sort_checkpoints_by_epoch_and_step(directory, data_type):
    # Get all checkpoints in the directory
    checkpoints = os.listdir(directory)
    
    # Define a regex pattern to extract epoch and step numbers
    pattern = r"test_{}_epoch_(\d+)_step_(\d+|end)\.pkl".format(data_type)
    
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


def read_checkpoints_and_gather_values(directory, data_type):
    sorted_checkpoints = sort_checkpoints_by_epoch_and_step(directory, data_type)
    
    stats = {'loss': [], 'accuracy': [], 'y_hats': [], 'ys': []}
    for checkpoint in sorted_checkpoints:
        with open(os.path.join(directory, checkpoint), 'rb') as f:
            data = pickle.load(f)
            for key in stats.keys():
                stats[key].append(data[key])
    
    return {key: np.array(value) for key, value in stats.items()}


def get_stats_over_seeds(args):
    stats = {'loss': [], 'accuracy': [], 'y_hats': [], 'ys': []}

    for seed in args.seeds:
        directory = os.path.join(args.save_path, args.dataset, f'{args.base_exp_name}_seed[{seed}]', 'stats')
        
        current_stats = read_checkpoints_and_gather_values(directory, args.data_type)
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
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})

    for category, entropies in category_entropies.items():
        mean_entropies = np.mean(entropies, axis=0)
        std_entropies = np.std(entropies, axis=0)
        plt.plot(mean_entropies, color=ALL_COLORS[category], label=ALL_LABELS[category], alpha=0.8)
        plt.fill_between(np.arange(len(mean_entropies)), mean_entropies - std_entropies, mean_entropies + std_entropies, alpha=0.2, color=ALL_COLORS[category])
    
    plt.xlabel('Steps', labelpad=10)
    plt.ylabel('Entropy', labelpad=20)
    plt.title('Entropy of Predictions', pad=10)
    plt.legend()
    plt.tight_layout()
    
    plt_name = f'results/entropy/{args.base_exp_name}_data_type[{args.data_type}]_entropy.png'
    plt.savefig(plt_name, dpi=300)
    plt.close()


def plot_categories(args, categories):
    
    # Plot percentage of points in each category
    total_points = len(categories['non_monotone']) + len(categories['easy']) + len(categories['hard']) + len(categories['monotone'])
    percentages = {key: len(categories[key]) * 100 / total_points for key in categories.keys()}
    print(percentages)
    # import ipdb; ipdb.set_trace()

    plt.figure(figsize=(8, 6))
    # Set the font size
    plt.rcParams.update({'font.size': 14})

    plt.bar(percentages.keys(), percentages.values(), color=[CATEGORY_COLORS[key] for key in percentages.keys()], alpha=0.5, edgecolor='black', width=0.5)
    # show percentages on top of bars
    for i, (_, value) in enumerate(percentages.items()):
        plt.text(i, value + 0.001, f'{value:.2f}%', ha='center', va='bottom', color='black')


    plt.xlabel('Category', labelpad=10)
    plt.ylabel('Percentage of Points (%)', labelpad=20)
    plt.title('Distribution of Points in Categories', pad=10)
    plt.ylim(0, max(percentages.values()) + 5)

    plt.xticks(list(CATEGORY_LABELS.keys()), CATEGORY_LABELS.values())
    plt.tight_layout()

    plt_name = f'results/category/{args.base_exp_name}_data_type[{args.data_type}]_categories.png'
    plt.savefig(plt_name, dpi=300)
        

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
    # Save stats
    pickle.dump(stats, open(f'results/stats/{args.base_exp_name}.pkl', 'wb'))

    # Select subset of stats
    stats = select_subset(stats, 0, max_steps)

    # Define categories of points
    avg_stats = get_avg_stats_over_seeds(stats)
    categories = get_point_categories(avg_stats, threshold=3.0)

    # Plot categories
    plot_categories(args, categories)

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
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use', choices=['cifar10', 'cifar10s'])
    parser.add_argument('--data_type', default='clean', type=str, help='Data type to use', choices=['clean', 'adv'])
    parser.add_argument('--seeds', nargs='+', type=int, help='List of seeds')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
