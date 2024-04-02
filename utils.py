import os
import yaml
import json
import numpy as np

import torch
import torch.optim as optim

from torchattacks import PGD

from models.resnet import ResNet18
from trainer import NT, AT, NRT, ART


def seed_experiment(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def get_model(dataset, model):
    if 'cifar10' == dataset:
        if model == 'resnet18':
            return ResNet18()
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
                       eps=args.attack_eps,
                       steps=args.attack_steps,
                       alpha=args.attack_step_size,
                       random_start=True)
        else:
            raise NotImplementedError(f'Attack name not recognized {args.attack_name}!')

    else:
        raise NotImplementedError(f'Dataset not recognized {args.attack_name}!')


def get_trainer(args, model, opt, device, adversary):
    if args.trainer == 'nt':
        return NT(args=args, model=model, opt=opt, device=device)
    elif args.trainer == 'nrt':
        return NRT(args=args, model=model, opt=opt, device=device)
    elif args.trainer == 'at':
        return AT(args=args, model=model, opt=opt, device=device, adversary=adversary)
    elif args.trainer == 'art':
        return ART(args=args, model=model, opt=opt, device=device, adversary=adversary)
    else:
        raise ValueError(f'Invalid trainer {args.trainer}!')


def get_exp_name(args):
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
    if args.trainer in ('at', 'art'):
        name = f"{name}_attack[{args.attack_name}]"

    # General
    if args.exp_spec:
        name = f"{name}_exp_spec[{args.exp_spec}]"
    name = f"{name}_seed[{args.seed}]"
    return name


def create_safe_path(path):
    postfix = 1
    safe_path = path
    while os.path.exists(safe_path):
        safe_path = path + f'_{postfix}'
        postfix += 1

    os.makedirs(safe_path)

    return safe_path


def get_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


def save_args(args, trained_model_path):
    save_path = os.path.join(trained_model_path, 'args.json')
    with open(save_path, 'w') as file:
        json.dump(vars(args), file)
