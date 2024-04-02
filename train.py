import os
import argparse

import torch
import torch.nn as nn

import wandb

from utils import seed_experiment, get_exp_name, create_safe_path, save_args
from utils import get_model, get_optimizer, get_lr_scheduler, get_adversary, get_trainer

import dataset
from dataset import get_data_loaders


def main(args):
    # Seed the experiment, for repeatability
    seed_experiment(args.seed)

    # Use GPUs if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    # Define the experiment
    exp_name = get_exp_name(args=args)

    # Define the trained model path
    trained_model_path = os.path.join(args.save_path, f'checkpoints/{args.dataset}', exp_name)
    trained_model_path = create_safe_path(trained_model_path)
    safe_exp_name = os.path.split(trained_model_path)[-1]

    # Save arguments for reproducibility
    save_args(args=args, trained_model_path=trained_model_path)

    # Init WandB
    wandb.init(name=safe_exp_name, entity=args.wandb_entity_name, project=args.wandb_project_name,
               config={'args': vars(args)}, tags=[args.trainer, args.model, args.dataset])

    # Define a name for the model's checkpoint
    model_filename = f"{args.trainer}_{args.dataset}_{args.model}"

    # Load the dataset
    data_path = os.path.join(args.data_path, args.dataset)
    dataset_norm = True if args.dataset_norm == 'on' else False
    train_loader, test_loader = get_data_loaders(args=args, data_path=data_path, norm=dataset_norm)

    # Init the model
    model = get_model(dataset=args.dataset, model=args.model).to(device)
    wandb.watch(model)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Setup optimizer
    opt = get_optimizer(args=args, model=model)
    lr_scheduler = get_lr_scheduler(args=args, opt=opt)

    # Define adversary
    adversary = get_adversary(args=args, model=model)
    if dataset_norm:
        adversary.set_normalization_used(mean=dataset.CIFAR_MEAN, std=dataset.CIFAR_STD)

    # Get the trainer
    trainer = get_trainer(args=args, model=model, opt=opt, device=device, adversary=adversary)

    for epoch in range(args.epochs):
        # Log learning rate
        wandb.log({'Learning rate': lr_scheduler.get_last_lr()[0], 'epoch': epoch})

        trainer.train(loader=train_loader, epoch=epoch)

        # plot histogram of weights and gradient:
        trainer.log_model_params(epoch)

        # Adjust learning rate for SGD
        lr_scheduler.step()

        # Evaluation on the train clean examples
        trainer.evaluate(loader=train_loader, adversary=None, epoch=epoch)

        # Evaluation on test clean examples
        trainer.evaluate(loader=test_loader, adversary=None, epoch=epoch)

        # Evaluation on test adversarial examples
        trainer.evaluate(loader=test_loader, adversary=adversary, epoch=epoch)

        # Save the checkpoint
        torch.save(model.state_dict(), os.path.join(trained_model_path, model_filename + f'_{epoch}.pt'))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Adversarial/Natural Training')
    parser.add_argument('--exp_name', default='', type=str, help='If None, args are combined for the name')
    parser.add_argument('--seed', default=42, type=int, help='Seed for reproducibility.')
    parser.add_argument('--exp_spec', default='', type=str, help='Specific experiment name.')

    # Run
    run_args = parser.add_argument_group('Run')
    run_args.add_argument('--trainer', default='nt', type=str, choices=['nt', 'at'])
    run_args.add_argument('--model', default='resnet18', type=str, help='Choose the model.', choices=['resnet18'])
    run_args.add_argument('--epochs', default=120, type=int, help='Number of epochs.')
    run_args.add_argument('--batch_size', default=128, type=int, help='Batch size.')

    # Dataset
    data_args = parser.add_argument_group('Dataset')
    data_args.add_argument('--data_path', default='./data', type=str, help='Directory path to data.')
    data_args.add_argument('--dataset', default='cifar10', type=str, help='Dataset to choose.', choices=['cifar10'])
    data_args.add_argument('--dataset_norm', default='off', type=str, help='Normalize dataset.', choices=['off', 'on'])

    # Optimizer
    opt_args = parser.add_argument_group('Optimizer')
    opt_args.add_argument('--opt_type', default='sgd', type=str, help='Choose optimizer.', choices=['sgd', 'adamw'])
    opt_args.add_argument('--lr', default=0.1, type=float, help='Learning rate.')
    opt_args.add_argument('--wd', default=1e-4, type=float, help='Weight decay.')
    opt_args.add_argument('--momentum', default=0.9, type=float, help='Momentum.')

    opt_args.add_argument('--lr_scheduler', default='multi', type=str, help='LR Scheduler', choices=['multi', 'cosine'])
    opt_args.add_argument('--gamma', default=0.1, type=float, help='Gamma of the multi lr scheduler.')
    opt_args.add_argument('--stones', default=[70, 90], nargs='+', type=float, help='Stones of the multi lr scheduler.')

    # Attack
    attack_args = parser.add_argument_group('Attack')
    attack_args.add_argument('--attack_name', default='pgd', type=str, help='Attack to perform.', choices=['pgd'])
    attack_args.add_argument('--attack_norm', default='l_inf', type=str, help='Norm of the attack.', choices=['l_inf'])
    attack_args.add_argument('--attack_eps', default=0.031, type=float, help='Epsilon of the attack.')
    attack_args.add_argument('--attack_steps', default=10, type=int, help='Steps of the attack.')
    attack_args.add_argument('--attack_step_size', default=0.007, type=float, help='Step size of the attack.')

    # Logging
    log_args = parser.add_argument_group('Logging')
    log_args.add_argument('--log_interval', default=100, type=int, help='Batches to wait before logging.')
    log_args.add_argument('--save_path', default='./logs', type=str, help='Path to save checkpoints')
    log_args.add_argument('--wandb_entity_name', default='entity_name', type=str, help='WandB entity name.')
    log_args.add_argument('--wandb_project_name', default='project_name', type=str, help='WandB project name.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
