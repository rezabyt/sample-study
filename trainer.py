import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F

import wandb


class NT:
    def __init__(self, args, model, opt, device):
        self.args = args
        self.model = model
        self.opt = opt
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

    def train(self, loader, epoch):
        self.model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(loader):
            loss = self.update(batch=batch, epoch=epoch, batch_idx=batch_idx, loader_length=len(loader.dataset))
            epoch_loss += loss.item() / len(loader)

        wandb.log({'Train Loss': epoch_loss, 'epoch': epoch})
        return epoch_loss

    def evaluate(self, loader, adversary, epoch):
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

        loss /= len(loader)
        accuracy = correct / len(loader.dataset)

        mode = 'ADV' if adversary else 'CLN'
        print(f'{loader.name.upper()} {mode}: Loss: {loss:.5f}, Accuracy: {100. * accuracy:2.2f}%')

        loss_key_name = f'{loader.name.upper()} {mode} Loss'
        wandb.log({loss_key_name: loss, 'epoch': epoch})

        acc_key_name = f'{loader.name.upper()} {mode} Accuracy'
        wandb.log({acc_key_name: accuracy, 'epoch': epoch})

        return {'epoch': epoch, 'loss': loss, 'accuracy': accuracy, 'y_hats': np.array(y_hats), 'ys': np.array(ys)}
    
    def eval(self, exp_directory, train_loader, test_loader, epoch, adversary=None):
         # Evaluation on the train clean examples
        train_clean_stats = self.evaluate(loader=train_loader, adversary=None, epoch=epoch)
        self.save_data_stats(exp_directory, train_clean_stats, epoch, 'train_clean')

        # Evaluation on test clean examples
        test_clean_stats = self.evaluate(loader=test_loader, adversary=None, epoch=epoch)
        self.save_data_stats(exp_directory, test_clean_stats, epoch, 'test_clean')
        

    def save_data_stats(self, exp_directory, stats, epoch, mode):
        stat_path = os.path.join(exp_directory, 'stats', mode, f'epoch_{epoch}.pkl')

        # Save the stats to a file
        with open(stat_path, 'wb') as file:
            pickle.dump(stats, file)


    def log_model_params(self, epoch):
        for name, param in self.model.named_parameters():
            gradients = wandb.Histogram(param.grad.data.cpu().detach().numpy())
            weights = wandb.Histogram(param.data.cpu().detach().numpy())

            wandb.log({'epoch': epoch, f'Gradients/{name}': gradients, f'Weights/{name}': weights})


class AT(NT):
    def __init__(self, args, model, opt, device, adversary):
        super(AT, self).__init__(args, model, opt, device)
        self.adversary = adversary

    def update(self, batch, epoch, batch_idx, loader_length):
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)

        X_adv = self.adversary(X, y)
        batch = (X_adv, y)
        return super().update(batch, epoch, batch_idx, loader_length)

    def eval(self, exp_directory, train_loader, test_loader, epoch, adversary):
        # Evaluation on the train clean examples
        train_clean_stats = self.evaluate(loader=train_loader, adversary=None, epoch=epoch)
        self.save_data_stats(exp_directory, train_clean_stats, epoch, 'train_clean')

        # Evaluation on test clean examples
        test_clean_stats = self.evaluate(loader=test_loader, adversary=None, epoch=epoch)
        self.save_data_stats(exp_directory, test_clean_stats, epoch, 'test_clean')

        # Evaluation on test adversarial examples
        test_adv_stats = self.evaluate(loader=test_loader, adversary=adversary, epoch=epoch)
        self.save_data_stats(exp_directory, test_adv_stats, epoch, 'test_adv')