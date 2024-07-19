# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.

import argparse
import datetime
import os
import time

# from architectures import ARCHITECTURES, get_architecture
from datasets import get_dataset, DATASETS
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import numpy as np
from models import *

parser = argparse.ArgumentParser(description='PyTorch Randomized Smoothing Training')

# dataset
parser.add_argument('--dataset', type=str, default ='cifar10', choices=DATASETS)
parser.add_argument('--batch', default=256, type=int, metavar='N', help='batchsize (default: 256)')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

# optimizer
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')

# save path
parser.add_argument('--model_dir', type=str, default='trained_models/', help='folder to save model and training log)')

# specify model
parser.add_argument('--mode', type=str, default='normal', help='full/partia/normal')

# specify training
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30, help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--seed', type=int, default=2, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
                    
# randomized smoothing
parser.add_argument('--noise_sd', default=0.0, type=float, help="standard deviation of Gaussian noise for data augmentation")

parser.add_argument('--gpu', default=None, type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()


def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # save_path
    training_speficiation = args.mode + '_std_' + str(args.noise_sd) + "_lr_" + str(args.lr) + '_seed_' + str(args.seed)
    save_dir = os.path.join(args.model_dir, args.dataset, args.mode, training_speficiation)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("we are saving model to: ", save_dir)

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)
    
    torch.manual_seed(args.seed)

            
    model.cuda()

    criterion = CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    for epoch in range(args.epochs):
        scheduler.step(epoch)
        before = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.noise_sd)
        test_loss, test_acc = test(test_loader, model, criterion, args.noise_sd)
        after = time.time()

        print('EVAL Epoch: {} \tLoss: {:.3f}\Acc: {:.3f}%'.format(
                epoch, test_loss, test_acc*100))

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(save_dir, 'checkpoint.pth.tar'))


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float):
    # switch to train mode
    model.train()
    loss_his = []
    acc_his = []
    num_batches = len(loader)
    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # augment inputs with noise
        inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

        outputs, predictions = model(inputs)

        loss = criterion(outputs, targets)
        acc = torch.mean((predictions==targets).float())

        loss_his.append(loss.item())
        acc_his.append(acc.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print progress
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.3f}\tAcc: {:.3f}%'.format(
                epoch, i, num_batches, loss.item(), acc.item()*100))

    return np.mean(loss_his), np.mean(acc_his)


def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
    # switch to eval mode
    model.eval()
    loss_his = []
    acc_his = []
    num_batches = len(loader)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            # compute output
            outputs, predictions = model(inputs)

            loss = criterion(outputs, targets)
            acc = torch.mean((predictions==targets).float())

            loss_his.append(loss.item())
            acc_his.append(acc.item())

        return np.mean(loss_his), np.mean(acc_his)


if __name__ == "__main__":
    main()
