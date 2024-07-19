import argparse
import copy
import logging
import os
import time
import math
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

from robust_net import LipNet
from modules import *

from utils import (upper_limit, lower_limit, cifar10_mean, cifar10_std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard, evaluate_certificates)

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'cifar10', type = str, 
                        choices = ['cifar10', 'cifar100'], 
                        help = 'The dataset for training.')
    parser.add_argument('--data-dir', default = 'data', type = str, 
                        help = 'The data directory.')
    parser.add_argument('--out-dir',  default = 'outputs', type = str,
                        help = 'The output directory.')
    parser.add_argument('--name', default = 'default', type = str,
                        help = 'The model name.')

    parser.add_argument('--image-size', default = 32, type = int,
                        help = 'The image size.')

    parser.add_argument('--opt-level', default = 'O1', type = str, choices = ['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')

    parser.add_argument('--seed', default = 0, type = int, help = 'Random seed')
    parser.add_argument('--epochs', default = 200, type = int)
    parser.add_argument('--batch-size', default = 256, type = int) # default = 128

    parser.add_argument('--optimizer', default = 'SGD', type = str, choices = ['SGD', 'Adam'])
    parser.add_argument('--loss-scale', default = '1.0', type = str, choices=['1.0', 'dynamic'],
        help = 'If loss_scale is "dynamic", adaptively adjust the loss scale over time.')
    parser.add_argument('--lr-schedule', default = 'multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default = 0.,  type = float)
    parser.add_argument('--lr-max', default = 0.1, type = float)
    parser.add_argument('--weight-decay', default = 5e-4, type = float)
    parser.add_argument('--momentum', default = 0.9, type = float)
    parser.add_argument('--epsilon', default = 36, type = int)
    parser.add_argument('--base-channels', default = 32, type = int)
    parser.add_argument('--num-layers', default = 10, type = int, help = 'model type') # default = 20

    parser.add_argument('--conv', default = 'BjorckConv', type = str,
                        choices = ['nn.Conv2d', 'SCFac', 'BCOP', 'BjorckConv', 'CayleyConv', 'LieExpConv'],
                        help = 'The type of convolutional layer used in the Lipschitz network.')
    parser.add_argument('--linear', default = 'BjorckLinear', type = str,
                        choices = ['nn.Linear', 'BjorckLinear', 'CayleyLinear', 'LieExpLinear'],
                        help = 'The type of linear layer used in the Lipschitz network.')
    parser.add_argument('--kernel-size', default = 3, type = int) 

    parser.add_argument('--power-thres', default = 1e-6, type = float)
    parser.add_argument('--bjorck-thres', default = 1e-3, type = float)

    return parser.parse_args()


def main():
    args = get_args()

    if args.dataset not in ['cifar10', 'cifar100']:
        raise Exception('Unknown dataset ', args.dataset)
    
    os.makedirs(args.out_dir, exist_ok = True)

    args.out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(args.out_dir, exist_ok = True)

    if args.name == 'default':
        args.name = 'LipNet%d_%s_%s' % (args.num_layers, args.conv, args.lr_max)

    args.out_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(args.out_dir, exist_ok = True)

    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.dataset, args.image_size)
    std = cifar10_std
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        raise Exception('Unknown dataset')

    conv = eval(args.conv)
    linear = eval(args.linear)

    if conv is BjorckConv:
        conv = functools.partial(conv, power_thres = args.power_thres, bjorck_thres = args.bjorck_thres)

    if linear is BjorckLinear:
        linear = functools.partial(linear, power_thres = args.power_thres, bjorck_thres = args.bjorck_thres)

    model = LipNet(conv, linear, args.kernel_size,
        in_shape = args.image_size, base_channels = args.base_channels,
        num_layers = args.num_layers, num_classes = num_classes).cuda()
    model.train()
    
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr = args.lr_max)
    elif args.optimizer == 'SGD':
        if args.conv == 'LieExpConv' and args.conv == 'LieExpLinear':
            opt = torch.optim.SGD(model.parameters(), lr = args.lr_max, momentum = args.momentum, 
                                  weight_decay = args.weight_decay)
        else: # if args.conv != 'LipExpConv' or args.conv == 'LieExpLinear':
            opt = torch.optim.SGD(model.parameters(), lr = args.lr_max, momentum = args.momentum,
                                  weight_decay = 0.)
        
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = True
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        if args.optimizer == 'Adam':
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                step_size_up=lr_steps / 20, step_size_down= (3 * lr_steps) / 20, cycle_momentum = False)
        elif args.optimizer == 'SGD':
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                step_size_up=lr_steps / 20, step_size_down= (3 * lr_steps) / 20)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps // 2, 
            (3 * lr_steps) // 4], gamma=0.1)

    # Training
    std = torch.tensor(std).cuda()
    L = 1/torch.max(std)
    prev_test_acc = 0.
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t Train Loss \t Train Acc \t Train Robust \t ' + 
                'Train Cert \t Test Loss \t Test Acc \t Test Robust \t Test Cert')
    for epoch in range(args.epochs):
        model.train()
        start_epoch_time = time.time()
        train_loss = 0
        train_cert = 0
        train_robust = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            
            output = model(X)
            output_y = output[torch.arange(X.shape[0]), y]
            
            onehot_y = torch.zeros_like(output).cuda()
            onehot_y[torch.arange(output.shape[0]), y] = 1.
            output_trunc = output - onehot_y*1e3
            
            output_runner_up, _ = torch.max(output_trunc, dim=1)
            margin = output_y - output_runner_up
            ce_loss = criterion(output, y)

            curr_cert = margin/(math.sqrt(2)*L)
            loss = ce_loss
            print(loss.data)

            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()

            curr_correct = (output.max(1)[1] == y)
            train_loss += ce_loss.item() * y.size(0)
            train_cert += (curr_cert * curr_correct).sum().item()
            train_robust += ((curr_cert > (args.epsilon/255.)) * curr_correct).sum().item()
            train_acc += curr_correct.sum().item()
            train_n += y.size(0)
            scheduler.step()
            
        last_state_dict = copy.deepcopy(model.state_dict())
            
        # Check current test accuracy of model
        test_loss, test_acc, mean_cert, robust_acc = evaluate_certificates(test_loader, model, L)
        if (robust_acc > prev_test_acc):
            model_path = os.path.join(args.out_dir, 'best.pth')
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, model_path)
            prev_test_acc = robust_acc
            best_epoch = epoch
        
        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n, 
            train_robust/train_n, train_cert/train_acc, test_loss, test_acc, robust_acc, mean_cert)
        
        model_path = os.path.join(args.out_dir, 'last.pth')
        torch.save(last_state_dict, model_path)
        
        trainer_state_dict = { 'epoch': epoch, 'optimizer_state_dict': opt.state_dict()}
        opt_path = os.path.join(args.out_dir, 'last_opt.pth')
        torch.save(trainer_state_dict, opt_path)          
        
        print("Finish epoch %d." % epoch)

    train_time = time.time()

    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)
    
    start_test_time = time.time()
    # Evaluation at early stopping
    model_test = LipNet(conv, linear, args.kernel_size,
        in_shape = args.image_size, base_channels = args.base_channels,
        num_layers = args.num_layers, num_classes = num_classes).cuda()
    model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()
        
    start_test_time = time.time()
    test_loss, test_acc, mean_cert, robust_acc = evaluate_certificates(test_loader, model_test, L)
    total_time = time.time() - start_test_time
    
    logger.info('Best Epoch \t Test Loss \t Test Acc \t Robust Acc \t Mean Cert \t Test Time')
    logger.info('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f', best_epoch, test_loss, 
                                                      test_acc, robust_acc, 
                                                      mean_cert, total_time)

    # Evaluation at last model
    model_test.load_state_dict(last_state_dict)
    model_test.float()
    model_test.eval()

    start_test_time = time.time()
    test_loss, test_acc, mean_cert, robust_acc = evaluate_certificates(test_loader, model_test, L)
    total_time = time.time() - start_test_time
    
    logger.info('Last Epoch \t Test Loss \t Test Acc \t Robust Acc \t Mean Cert \t Test Time')
    logger.info('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f', epoch, test_loss, 
                                                        test_acc, robust_acc, 
                                                        mean_cert, total_time)

if __name__ == "__main__":
    main()
