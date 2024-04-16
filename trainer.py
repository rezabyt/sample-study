import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F

import wandb


class NT:
    def __init__(self, args, exp_directory, train_loader, test_loader, model, opt, lr_scheduler, device):
        self.args = args
        self.exp_directory = exp_directory
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.opt = opt
        self.lr_scheduler = lr_scheduler
        self.device = device

    def get_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def update(self, batch, epoch, batch_idx, loader_length):
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)

        self.opt.zero_grad()
        y_hat = self.model(X)
        loss = self.get_loss(y_hat, y)
        loss.backward()
        self.opt.step()

        # Print progress
        if batch_idx % self.args.log_interval == 0:
            print(
                f'Epoch: {epoch}/{self.args.epochs} '
                f'Progress: [{batch_idx * len(X)}/{loader_length} ({100. * batch_idx * len(X) / loader_length:.0f}%)] '
                f'Loss: {loss.item():.6f}')
        return loss

    def train(self, epoch):
        self.model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.update(batch=batch, epoch=epoch, batch_idx=batch_idx, loader_length=len(self.train_loader.dataset))
            epoch_loss += loss.item() / len(self.train_loader)
            
            base_epoch = self.args.epochs // 10
            num_batches = len(self.train_loader)
            if batch_idx % ((epoch + 1) * (num_batches // base_epoch)) == 0 and batch_idx != 0:
                self.eval(epoch=epoch, step=batch_idx)
                self.save_model(epoch=epoch, step=batch_idx)

        wandb.log({'Train Loss': epoch_loss, 'epoch': epoch})

        # Adjust learning rate
        self.lr_scheduler.step()

        return epoch_loss

    def evaluate(self, loader, epoch, step, adversary):
        self.model.eval()

        loss = 0
        correct = 0
        y_hats = []
        ys = []
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)

            if adversary:
                X = adversary(X, y)

            with torch.no_grad():
                y_hat = self.model(X)
            
            # Append the predictions and true labels for later use
            y_hats.extend(y_hat.cpu().numpy())
            ys.extend(y.cpu().numpy())

            loss += self.get_loss(y_hat, y).item()
            pred = y_hat.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
        
        self.model.train()

        loss /= len(loader)
        accuracy = correct / len(loader.dataset)

        if step == 'end':
            mode = 'ADV' if adversary else 'CLN'
            print(f'{loader.name.upper()} {mode}: Loss: {loss:.5f}, Accuracy: {100. * accuracy:2.2f}%')

            loss_key_name = f'{loader.name.upper()} {mode} Loss'
            wandb.log({loss_key_name: loss, 'epoch': epoch})

            acc_key_name = f'{loader.name.upper()} {mode} Accuracy'
            wandb.log({acc_key_name: accuracy, 'epoch': epoch})

        return {'epoch': epoch, 'step': step, 'loss': loss, 'accuracy': accuracy, 'y_hats': np.array(y_hats), 'ys': np.array(ys)}
    
    def eval(self, epoch, step='end'):
        # Evaluation on the train clean examples
        self.evaluate(loader=self.train_loader, epoch=epoch, step=step, adversary=None)

        # Evaluation on test clean examples and save the stats
        test_clean_stats = self.evaluate(loader=self.test_loader, epoch=epoch, step=step, adversary=None)
        self.save_data_stats(test_clean_stats, epoch, step, 'test_clean')
        

    def save_data_stats(self, stats, epoch, step, mode):
        stat_path = os.path.join(self.exp_directory, 'stats', f'{mode}_epoch_{epoch}_step_{step}.pkl')

        # Save the stats to a file
        with open(stat_path, 'wb') as file:
            pickle.dump(stats, file)


    def log_learning_rate(self, epoch):
        wandb.log({'Learning rate': self.lr_scheduler.get_last_lr()[0], 'epoch': epoch})


    def log_model_params(self, epoch):
        for name, param in self.model.named_parameters():
            gradients = wandb.Histogram(param.grad.data.cpu().detach().numpy())
            weights = wandb.Histogram(param.data.cpu().detach().numpy())

            wandb.log({'epoch': epoch, f'Gradients/{name}': gradients, f'Weights/{name}': weights})
    
    def save_model(self, epoch, step='end'):
        # Save model and optimizer states
        checkpoint_path = os.path.join(self.exp_directory, 'checkpoints', f'epoch_{epoch}_step_{step}.pt')
        state_dicts = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict()}
        
        torch.save(state_dicts, checkpoint_path)
    
    def load_from_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        return checkpoint['epoch'] + 1

class AT(NT):
    def __init__(self, args, exp_directory, train_loader, test_loader, model, opt, lr_scheduler, device, adversary):
        super(AT, self).__init__(args, exp_directory, train_loader, test_loader, model, opt, lr_scheduler, device)
        self.adversary = adversary

    def update(self, batch, epoch, batch_idx, loader_length):
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)

        X_adv = self.adversary(X, y)
        batch = (X_adv, y)
        return super().update(batch, epoch, batch_idx, loader_length)

    def eval(self, epoch, step='end'):
        # Evaluation on the train clean examples
        self.evaluate(loader=self.train_loader, epoch=epoch, step=step, adversary=None)

        # Evaluation on test clean examples and save the stats
        test_clean_stats = self.evaluate(loader=self.test_loader, epoch=epoch, step=step, adversary=None)
        self.save_data_stats(test_clean_stats, epoch, step, 'test_clean')

        # Evaluation on test adversarial examples and save the stats
        test_adv_stats = self.evaluate(loader=self.test_loader, epoch=epoch, step=step, adversary=self.adversary)
        self.save_data_stats(test_adv_stats, epoch, step, 'test_adv')