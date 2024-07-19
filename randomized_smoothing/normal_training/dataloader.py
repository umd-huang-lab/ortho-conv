import numpy as np
import torch 
import torchvision
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

class MyDataset(Dataset):
  def __init__(self, database='MNIST'):
    super().__init__()
    self.database = database
    
    if self.database == 'MNIST':
        transform = transforms.Compose(
                                [transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])

        self.trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                            download=True, transform=transform)
        
        num_train = len(self.trainset)
        indices = list(range(num_train))
        split = int(np.floor(0.2*num_train))
        
        self.valset = torch.utils.data.Subset(self.trainset, indices[:split])

        self.trainset = torch.utils.data.Subset(self.trainset, indices[split:])

        self.testset = torchvision.datasets.MNIST(root='../data', train=False,
                                            download=True, transform=transform)

    elif self.database == 'CIFAR10':
        transform = transforms.Compose(
                                [transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        self.trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
        
        num_train = len(self.trainset)
        indices = list(range(num_train))
        split = int(np.floor(0.2*num_train))
        
        self.valset = torch.utils.data.Subset(self.trainset, indices[:split])

        self.trainset = torch.utils.data.Subset(self.trainset, indices[split:])

        self.testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=False, transform=transform)

    else: # this means there is no database available, it is wrong
        self.trainset = None
        self.testset = None

def get_loader(database='MNIST', flag='train', batch_size=4, shuffle=False, num_workers=2):

    dataset = MyDataset(database)

    if flag == 'train':
        train_loader = torch.utils.data.DataLoader(dataset.trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(dataset.valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader

    elif flag == 'test':
        test_loader = torch.utils.data.DataLoader(dataset.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return test_loader

if __name__ == "__main__":

    train_loader, val_loader = get_loader(flag='train', batch_size=16, num_workers=4, shuffle=False)
    print(len(train_loader))
    print(len(val_loader))

    for i, (inputs,targets) in enumerate(train_loader):
        # print(inputs.shape)
        print("train: ", targets)
        break

    for i, (inputs, targets) in enumerate(val_loader):
        # print(inputs.shape)
        print("val: ", targets)
        break