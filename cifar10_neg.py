import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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


if __name__ == '__main__':
    # Load the CIFAR10-neg dataset
    cifar10_neg_loader = get_cifar10_neg_loader()
    for i, (imgs, labels) in enumerate(cifar10_neg_loader):
        print(imgs.shape, labels.shape)
