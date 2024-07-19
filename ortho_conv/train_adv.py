import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os
import utils

# how to change the batch size:
# os.environ['batch_size'] = '256'
import data

from networks.models import *
from networks.lipnet import LipNet
from networks.resnet import ResNet, WideResNet
from networks.convnet import ConvNet, WideConvNet
from networks.shufflenet import ShuffleNet, WideShuffleNet

from networks.resnet_ import WideResNet as WideResNetN

parser = argparse.ArgumentParser()
parser.add_argument('--model', default = 'KWLarge',
                    choices=['KWLarge', 'ResNet9', 'WideResNetC', 'LipNet',
                    'ShuffleNet', 'ResNet', 'WideResNet', 'WideResNetN'])
parser.add_argument('--layers', default = 10, type = int) # for ShuffleNet, ResNet, WideResNet, and WideResNetC
parser.add_argument('--factor', default =  1, type = int) # for WideResNet and WideResNetC
parser.add_argument('--downsampling', default = 'pool',
                    choices = ['pool', 'stride_wide', 'stride_slim'])

parser.add_argument('--kernel', default =  3, type = int)
parser.add_argument('--dilate', default =  1, type = int)
parser.add_argument('--init', default = 'permutation', type = str)

parser.add_argument('--linear', default = 'CayleyLinear',
                    choices = ['nn.Linear', 'BjorckLinear', 'CayleyLinear', 'LieExpLinear'])
parser.add_argument('--conv',   default = 'CayleyConv',
                    choices = ['PlainConv', 'SVCM', 'BCOP', 'CayleyConv', 'LieExpConv',
                    'Paraunitary', 'SCFac', 'RKO',  'OSSN'])

parser.add_argument('--norm', default = 'None', choices = ['None', 'nn.BatchNorm2d'])
parser.add_argument('--actv', default = 'GroupSort', choices = ['GroupSort', 'nn.ReLU'])

parser.add_argument('--path', default = './checkpoints', type = str)
parser.add_argument('--name', default = 'default', type = str)

parser.add_argument('--epochs', default = 200, type = int)
parser.add_argument('--lr_max', default = 0.01, type = float)

parser.add_argument('--ce_loss', action = 'store_true')
parser.add_argument('--stddev', action = 'store_true')

parser.add_argument('--eps_0', default = 0.5, type = float) # for training
parser.add_argument('--alpha_0', default = 10, type = float)

parser.add_argument('--eps',  default = 8.0, type = float) # for evaluation
parser.add_argument('--alpha', default = 2.0, type = float)

args = parser.parse_args()

eps = args.eps / 255.0
alpha = args.alpha / 255.0
alpha_0 = args.alpha_0 / 255.0

print('stddev: ', args.stddev)
print('linear: ', args.linear)
print('conv: ', args.conv)
print('norm: ', args.norm)
print('actv: ', args.actv)


_model = eval(args.model)
_full = eval(args.linear)
_conv = eval(args.conv)
_norm = eval(args.norm) if args.norm != 'None' else None
_actv = eval(args.actv)

if args.model in ['KWLarge', 'ResNet9']:
    model_name = args.model

    model = _model(conv = _conv, linear = _full)
    
elif args.model in ['WideResNetC']:
    model_name = args.model + str(args.layers) + '-' + str(args.factor)

    model = _model(conv = _conv, linear = _full, 
        depth = args.layers, widen_factor = args.factor,
        kernel_size = args.kernel, dilation = args.dilate, init = args.init)

elif args.model in ['LipNet']:
    model_name = args.model + str(args.layers)

    model = _model(conv = _conv, linear = _full, num_layers = args.layers)

elif args.model in ['ResNet', 'ShuffleNet', 'ConvNet']:
    model_name = args.model + str(args.layers)

    model = _model(conv = _conv, linear = _full, 
        num_layers = args.layers, downsampling = args.downsampling,
        conv_kernel = args.kernel, conv_dilate = args.dilate, init = args.init)

elif args.model in ['WideResNet', 'WideShuffleNet', 'WideConvNet']:
    model_name = args.model + str(args.layers) + '-' + str(args.factor)

    model = _model(conv = _conv, linear = _full, 
        num_layers = args.layers, widen_factor = args.factor, downsampling = args.downsampling,
        conv_kernel = args.kernel, conv_dilate = args.dilate, init = args.init)

elif args.model in ['WideResNetN']:
    model_name = args.model + str(args.layers) + '-' + str(args.factor)

    model = _model(conv = _conv, linear = _full, norm = _norm, actv = _actv,
        num_layers = args.layers, widen_factor = args.factor, downsampling = args.downsampling,
        conv_kernel = args.kernel)

print('model:  ', model_name)

if not os.path.isdir(args.path):
    os.mkdir(args.path)

if args.name == "default":
    args.name = "%s_%s_%s.pt" % (model_name, args.conv, args.eps_0)

model = nn.Sequential(
    Normalize(data.mu, data.std if args.stddev else 1.0),
    model
).cuda()

epochs = args.epochs
lr_max = args.lr_max

# for SVCM projections
proj_nits = 100

# lr schedule: superconvergence
lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, lr_max, lr_max/20.0, 0])[0]

# optimizer: Adam
opt = optim.Adam(model.parameters(), lr = lr_max, weight_decay = 0)

# loss: multi-margin loss
if args.ce_loss:
    criterion = lambda yhat, y: F.cross_entropy(yhat, y)
else:
    criterion = lambda yhat, y: utils.margin_loss(yhat, y, args.eps_0, 1.0, 1.0)

for epoch in range(epochs):
    start = time.time()
    train_loss, acc, n = 0, 0, 0
    for i, batch in enumerate(data.train_batches):
        lr = lr_schedule(epoch + (i + 1) / len(data.train_batches))
        opt.param_groups[0].update(lr = lr)

        X, y = batch['input'], batch['target']

        delta = torch.zeros_like(X)
        delta.uniform_(-eps, eps)
        delta.requires_grad = True
        output = model(X + delta)

        loss_ = criterion(output, y)
        loss_.backward()
        grad = delta.grad.detach()

        delta.data = utils.clamp(delta + alpha_0 * torch.sign(grad), -eps, eps)
        delta.data = utils.clamp(X + delta, 0, 1) - X
        delta = delta.detach()
        
        output = model(X + delta)
        loss = criterion(output, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss += loss.item() * y.size(0)
        acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)

        # for SVCM projections
        if i % proj_nits == 0 or i == len(data.train_batches) - 1:
            for m in model.modules():
                if hasattr(m, '_project'):
                    m._project()
    
    if (epoch + 1) % 10 == 0:
        l_emp = utils.empirical_local_lipschitzity(model, data.test_batches, early_stop=True).item()
        print('[{}] --- Empirical Lipschitzity: {}'.format(args.model, l_emp))

    print(f'[{args.model}] Epoch: {epoch} | Train Acc: {acc/n:.4f}, Test Acc: {utils.accuracy(model, data.test_batches):.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}')

torch.save(model.state_dict(), os.path.join(args.path, args.name))

# print('[{}] (EMPIRICAL) Robust accuracy (eps: {:.4f}): {}'.format(args.model, eps, utils.rob_acc(data.test_batches, model, eps, alpha, opt, False, 50, 10)[0]))
print('[{}] (EMPIRICAL) Robust accuracy (eps: {:.4f}): {}'.format(args.model, eps, utils.rob_acc(data.test_batches, model, eps, alpha, opt, False, 10, 1)[0]))         
