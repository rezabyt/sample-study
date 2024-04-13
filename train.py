import os
import argparse

import torch
import torch.nn as nn

import wandb

from utils import seed_experiment, get_exp_name, create_safe_path, save_args, get_last_checkpoint
from utils import get_model, get_optimizer, get_lr_scheduler, get_adversary, get_trainer
from utils import load_wandb_job_id, save_wandb_job_id

import dataset
from dataset import get_data_loaders

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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

    if args.resume == 'on':
        if not os.path.exists(trained_model_path):
            raise FileNotFoundError(f"Checkpoint path {trained_model_path} does not exist!")
        print(f"Resuming training from {trained_model_path}!")
        wandb_job_id = load_wandb_job_id(trained_model_path=trained_model_path)
    else:
        trained_model_path = create_safe_path(trained_model_path)
        wandb_job_id = wandb.util.generate_id()
        save_wandb_job_id(trained_model_path=trained_model_path, wandb_job_id=wandb_job_id)

    safe_exp_name = os.path.split(trained_model_path)[-1]

    # Save arguments for reproducibility
    save_args(args=args, trained_model_path=trained_model_path)

    # Init WandB
    wandb.init(name=safe_exp_name, entity=args.wandb_entity_name, project=args.wandb_project_name,
               config={'args': vars(args)}, tags=[args.trainer, args.model, args.dataset], 
               id=wandb_job_id, resume="allow")

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

    # Load the saved model, optimizer, and LR scheduler states if resuming
    if args.resume == 'on':
        checkpoint = get_last_checkpoint(trained_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        init_epoch = checkpoint['epoch'] + 1
    else:
        init_epoch = 0

    # Define adversary if trainer is AT
    if args.trainer == 'at':
        adversary = get_adversary(args=args, model=model)
        if dataset_norm:
            adversary.set_normalization_used(mean=dataset.CIFAR_MEAN, std=dataset.CIFAR_STD)
    else:
        adversary = None

    # Get the trainer
    trainer = get_trainer(args=args, model=model, opt=opt, device=device, adversary=adversary)

    for epoch in range(init_epoch, args.epochs):
        # Log learning rate
        wandb.log({'Learning rate': lr_scheduler.get_last_lr()[0], 'epoch': epoch})

        # Train the model for one epoch
        loss = trainer.train(loader=train_loader, epoch=epoch)

        # Plot histogram of weights and gradient:
        trainer.log_model_params(epoch)

        # Adjust learning rate
        lr_scheduler.step()
        
        # Evaluate the model on datasets
        trainer.eval(train_loader, test_loader, epoch, adversary)

        # Save model and optimizer states
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': loss}, os.path.join(trained_model_path, model_filename + f'_{epoch}.pt'))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Adversarial/Natural Training')
    parser.add_argument('--exp_name', default='', type=str, help='If None, args are combined for the name')
    parser.add_argument('--seed', default=42, type=int, help='Seed for reproducibility.')
    parser.add_argument('--exp_spec', default='', type=str, help='Specific experiment name.')
    parser.add_argument('--resume', default='off', type=str, help='Resume training from checkpoint.', choices=['off', 'on'])

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
