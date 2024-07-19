from comet_ml import Experiment
experiment = Experiment(api_key="wuXsTWjwWxv74mkYWwQbdlVOr",
                        project_name="CIFAR_normal", workspace="OrthoCNN",
                        parse_args=True)
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from models import *
from dataloader import *


parser = argparse.ArgumentParser(description='PyTorch Orthogonal Network Training')

# dataset
parser.add_argument('--database', type=str, default='CIFAR10', help='MNIST/CIFAR10')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=128, metavar='N', help='input batch size for testing (default: 128)')
# parser.add_argument('--shuffle', action="store_true", help='if nothing written, value=False, else True')
parser.add_argument('--shuffle', type=bool, default=True, help='if nothing written, value=False, else True')

# optimizer
parser.add_argument('--weight_decay','--wd',default=2e-4,type=float,metavar='W')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum')
     
# save path
parser.add_argument('--model_dir', default='trained_models/',help='directory of model for saving checkpoint') 

# specify training
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate')
parser.add_argument('--seed', type=int, default=2, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

####################################################################
########################### TRAINING ###############################
####################################################################

# save path 
training_speficiation = "preprocessor_resnet" "_lr_" + str(args.lr) + '_seed_' + str(args.seed) 
save_dir = os.path.join(args.model_dir, args.database, "normal", training_speficiation)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print("we are saving model to: ", save_dir)

experiment.set_name(training_speficiation)

# dataloader
train_loader, val_loader = get_loader(database=args.database, flag='train', batch_size=args.batch_size, num_workers=4, shuffle=False)

# Loss
criterion = nn.CrossEntropyLoss()

def train(args, preprocessor, model, train_loader, optimizer_preprocessor, optimizer_model, epoch):
    preprocessor.train()
    model.train()
    loss_his = []
    acc_his = []
    num_batches = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer_preprocessor.zero_grad()
        optimizer_model.zero_grad()

        z = preprocessor(inputs)
        outputs, predictions = model(z)

        loss = criterion(outputs, targets)
        acc = torch.mean((predictions==targets).float())

        loss.backward()
        optimizer_preprocessor.step()
        optimizer_model.step()

        loss_his.append(loss.item())
        acc_his.append(acc.item())

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.3f}\tAcc: {:.3f}%'.format(
                epoch, batch_idx, num_batches, loss.item(), acc.item()*100))

        # break

    return np.mean(loss_his), np.mean(acc_his)

def eval(args, preprocessor, model, val_loader, epoch):
    preprocessor.eval()
    model.eval()
    loss_his = []
    acc_his = []
    num_batches = len(val_loader)
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        z = preprocessor(inputs)
        outputs, predictions = model(z)

        loss = criterion(outputs, targets)
        acc = torch.mean((predictions==targets).float())

        loss_his.append(loss.item())
        acc_his.append(acc.item())
    
        # break

    return np.mean(loss_his), np.mean(acc_his)

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if epoch >= 40:
        lr = args.lr * 0.1
    if epoch >= 80:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # model
    torch.manual_seed(args.seed)

    # preprocessor
    preprocessor = Preprocessor()
    model = ResNet18(in_channels=64)
    preprocessor = preprocessor.cuda()
    model = model.cuda()

    params_preprocessor = list(preprocessor.parameters())
    params_model = list(model.parameters())

    optimizer_preprocessor = torch.optim.Adam(params_preprocessor, lr=args.lr)
    optimizer_model = torch.optim.Adam(params_model, lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
    best_acc = 0
    best_model = None
    best_epoch = 0
    best_optimizer = None

    for epoch in range(0, args.epochs):
        # adjust learning rate
        adjust_learning_rate(optimizer_preprocessor, epoch)
        adjust_learning_rate(optimizer_model, epoch)

        # training
        train_loss, train_acc = train(args, preprocessor, model, train_loader, optimizer_preprocessor, optimizer_model, epoch)

        if epoch%2 == 0:
            # validation
            val_loss, val_acc = eval(args, preprocessor, model, val_loader, epoch)
            print('EVAL Epoch: {} \tLoss: {:.3f}\Acc: {:.3f}%'.format(
                epoch, val_loss, val_acc*100))
            # save checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model.state_dict()
                best_epoch = epoch
                best_optimizer_preprocessor = optimizer_preprocessor.state_dict()
                best_optimizer_model = optimizer_model.state_dict()

            file_name = os.path.join(save_dir, "last.checkpoint")
            checkpoint = {'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer_preprocessor_state_dict': optimizer_preprocessor.state_dict(),
                        'optimizer_model_state_dict': optimizer_model.state_dict(),}

            if os.path.exists(file_name):
                print('Overwriting {}'.format(file_name))
            torch.save(checkpoint, file_name)

            curve_key = "acc train"
            experiment.log_metric(curve_key, train_acc*100, step=epoch)
            curve_key = "acc val"
            experiment.log_metric(curve_key, val_acc*100, step=epoch)
            curve_key = "loss train"
            experiment.log_metric(curve_key, train_loss, step=epoch)
            curve_key = "loss val"
            experiment.log_metric(curve_key, val_loss, step=epoch)

        # break 
    best_name = os.path.join(save_dir, "best.checkpoint")
    best_checkpoint = {'best_acc': best_acc,
                       'best_model':  best_model,
                       'best_epoch': best_epoch,
                       'best_optimizer_preprocessor': best_optimizer_preprocessor,
                       'best_optimizer_model': best_optimizer_model}

    torch.save(best_checkpoint, best_name)
    print("best acc: {:.3f}%".format(best_acc*100))
    print("training is over!!!!")

if __name__ == '__main__':
    main()