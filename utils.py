import os
import yaml
import json
import numpy as np

import torch
import torch.optim as optim

from torchattacks import PGD

from models.resnet import ResNet18
from trainer import NT, AT


def seed_experiment(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def get_model(dataset, model):
    if 'cifar' in dataset:
        num_classes = 10 if dataset in ('cifar10', 'cifar10s') else 100
        if model == 'resnet18':
            return ResNet18(num_classes=num_classes)
        else:
            raise ValueError(f'Model name not recognized ({model})')
    else:
        raise ValueError(f'Dataset not recognized ({dataset})')


def get_optimizer(args, model):
    if args.opt_type == 'sgd':
        opt = optim.SGD(params=model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    elif args.opt_type == 'adamw':
        opt = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError('Undefined optimizer {}'.format(args.opt_type))

    return opt


def get_lr_scheduler(args, opt):
    if args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=args.epochs)
    elif args.lr_scheduler == 'multi':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=opt, milestones=args.stones, gamma=args.gamma)
    else:
        raise ValueError('Invalid type for lr scheduler {}!'.format(args.lr_scheduler))
    return lr_scheduler


def get_adversary(args, model):
    if 'cifar10' in args.dataset:
        if args.attack_name == 'pgd' and args.attack_norm == 'l_inf':
            return PGD(model=model,
                       eps=args.attack_eps / 255.0,
                       steps=args.attack_steps,
                       alpha=args.attack_step_size / 255.0,
                       random_start=True)
        else:
            raise NotImplementedError(f'Attack name not recognized {args.attack_name}!')

    else:
        raise NotImplementedError(f'Dataset not recognized {args.attack_name}!')


def get_trainer(args, exp_directory, train_loader, test_loader, model, opt, lr_scheduler, device, adversary=None):
    if args.trainer == 'nt':
        return NT(args=args, exp_directory=exp_directory, train_loader=train_loader, test_loader=test_loader,
                   model=model, opt=opt, lr_scheduler=lr_scheduler, device=device)
    elif args.trainer == 'at':
        return AT(args=args, exp_directory=exp_directory, train_loader=train_loader, test_loader=test_loader, 
                  model=model, opt=opt, lr_scheduler=lr_scheduler, device=device, adversary=adversary)
    else:
        raise ValueError(f'Invalid trainer {args.trainer}!')


def get_exp_name(args, seed=None):
    if args.exp_name:
        return args.exp_name
    # Run
    name = f"trainer[{args.trainer}]_" \
           f"model[{args.model}]_" \
           f"epochs[{args.epochs}]_" \
           f"bs[{args.batch_size}]"

    # Dataset
    name = f"{name}_dataset[{args.dataset}]"

    # Optimizer
    name = f"{name}_opt[{args.opt_type}]_lr[{args.lr}]_lr_scheduler[{args.lr_scheduler}]"

    # Attack
    if args.trainer in ('at'):
        name = f"{name}_attack[{args.attack_name}]"

    # General
    if args.exp_spec:
        name = f"{name}_{args.exp_spec}"
    
    if seed:
        name = f"{name}_seed[{seed}]"
    else:
        name = f"{name}_seed[{args.seed}]"
    return name


def create_safe_path(exp_directory):
    postfix = 1
    safe_exp_directory = exp_directory
    while os.path.exists(safe_exp_directory):
        safe_exp_directory = exp_directory + f'_{postfix}'
        postfix += 1

    # Create main directory
    os.makedirs(safe_exp_directory)
    
    # Create subdirectories
    subdirectories = ['checkpoints', 'stats']
    for subdirectory in subdirectories:
        os.makedirs(os.path.join(safe_exp_directory, subdirectory))

    return safe_exp_directory


def get_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


def save_args(args, exp_directory):
    save_path = os.path.join(exp_directory, 'args.json')
    with open(save_path, 'w') as file:
        json.dump(vars(args), file)


def save_wandb_job_id(exp_directory, wandb_job_id):
    save_path = os.path.join(exp_directory, 'wandb_job_id.txt')
    with open(save_path, 'w') as file:
        file.write(wandb_job_id)


def load_wandb_job_id(exp_directory):
    save_path = os.path.join(exp_directory, 'wandb_job_id.txt')
    with open(save_path, 'r') as file:
        return file.read()
    

def get_last_checkpoint(exp_directory):
    checkpoints_directory = os.path.join(exp_directory, 'checkpoints')
    files = os.listdir(checkpoints_directory)
    checkpoint_files = [f for f in files if f.endswith('.pt') and 'end' in f]
    sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.split('_')[1]))
    return torch.load(os.path.join(checkpoints_directory, sorted_checkpoints[-1]))