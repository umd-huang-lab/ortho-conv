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

from autoattack import AutoAttack

from networks.models import *
from networks.lipnet import LipNet
from networks.resnet import ResNet, WideResNet
from networks.convnet import ConvNet, WideConvNet
from networks.shufflenet import ShuffleNet, WideShuffleNet

from resnet_ import WideResNet as WideResNetN

parser = argparse.ArgumentParser()
parser.add_argument('--model', default = 'KWLarge',
                    choices=['KWLarge', 'ResNet9', 'WideResNetC', 'LipNet',
                    'ShuffleNet', 'ResNet', 'WideResNet', 'WideResNetN', 'WideResNetN'])
parser.add_argument('--layers', default = 32, type = int) # for ShuffleNet, ResNet, WideResNet, and WideResNetC
parser.add_argument('--factor', default =  8, type = int) # for WideResNet and WideResNetC
parser.add_argument('--downsampling', default = 'pool',
                    choices = ['pool', 'stride_wide', 'stride_slim'])

parser.add_argument('--kernel', default =  3, type = int)
parser.add_argument('--dilate', default =  1, type = int)
parser.add_argument('--init', default = 'permutation', type = str)

parser.add_argument('--linear', default = 'CayleyLinear',
                    choices = ['nn.Linear', 'BjorckLinear', 'BjorckLinear_', 'CayleyLinear', 'LieExpLinear'])
parser.add_argument('--conv',   default = 'CayleyConv',
                    choices = ['PlainConv', 'BjorckConv', 'BjorckConv_', 'CayleyConv', 'LieExpConv',
                    'Paraunitary', 'SCFac', 'BCOP', 'RKO', 'SVCM', 'OSSN'])

parser.add_argument('--norm', default = 'None', choices = ['None', 'nn.BatchNorm2d'])
parser.add_argument('--actv', default = 'GroupSort', choices = ['GroupSort', 'nn.ReLU'])

parser.add_argument('--path', default = './checkpoints', type = str)
parser.add_argument('--name', default = 'default', type = str)

parser.add_argument('--stddev', action = 'store_true')
parser.add_argument('--eps_0', default = 0.5, type = float) # for training
parser.add_argument('--eps',  default = 36.0, type = float) # for evaluation

parser.add_argument('--autoattack', action = 'store_true')

args = parser.parse_args()

eps = args.eps / 255.0
alpha = eps / 4.0

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
        faconv_kernel = args.kernel)

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

# warm-up the model
with torch.no_grad(): 
    model(torch.randn(2, 3, 32, 32).cuda())

model.load_state_dict(torch.load(os.path.join(args.path, args.name)))

if args.model in ['WideResNet', 'WideResNetC', 'WideResNetN']:
    for name, params in model.named_parameters():
        if 'skip_connection.alpha' in name:
            print('%s: %3f'% (name, torch.sigmoid(params).data))

# loss: multi-margin loss
criterion = lambda yhat, y: utils.margin_loss(yhat, y, args.eps_0, 1.0, 1.0)
opt = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0)

# evalutate test mode
model.eval()
with torch.no_grad():
    test_acc = utils.accuracy(model, data.test_batches)
    test_mem = torch.cuda.max_memory_allocated()

# evaluate training mode
model.train()
start_time = time.time()
train_loss, acc, n = 0, 0, 0
for i, batch in enumerate(data.train_batches):
    X, y = batch['input'], batch['target']
            
    output = model(X)
    loss = criterion(output, y)

    opt.zero_grad()
    loss.backward()
    if i == 100:
        train_mem = torch.cuda.max_memory_allocated()
        
    train_loss += loss.item() * y.size(0)
    acc += (output.max(1)[1] == y).sum().item()
    n += y.size(0)

end_time = time.time()
train_acc = acc/n
train_time = end_time - start_time

print(f'[Train Acc: {train_acc:.4f}, Time: {train_time:.2f}, Memory: {train_mem:d}')

# evalutate test mode again
model.eval()
with torch.no_grad():
    start_time = time.time()
    test_acc = utils.accuracy(model, data.test_batches)
    end_time  = time.time()
    test_time = end_time - start_time

print(f'[Test Acc: {test_acc:.4f}, Time: {test_time:.2f}, Memory: {test_mem:d}')

# lipschitz constant
l_emp = utils.empirical_local_lipschitzity(model, data.test_batches, early_stop = True).item()
print('[{}] --- Empirical Lipschitzity: {}'.format(args.model, l_emp))

# certified accuracy
model.train()

if not args.stddev:
    print('[{}] (PROVABLE) Certifiably Robust (eps: {:.4f}): {:.4f}, Cert. Wrong: {:.4f}, Insc. Right: {:.4f}, Insc. Wrong: {:.4f}'.format(args.model, eps, *utils.cert_stats(model, data.test_batches, eps * 2**0.5, full=True)))

# robust accuracy (PGD)
print('[{}] (EMPIRICAL) Robust accuracy (eps: {:.4f}): {}'.format(args.model, eps, utils.rob_acc(data.test_batches, model, eps, alpha, opt, False, 10, 1, linf_proj=False, l2_grad_update=True)[0]))

if args.autoattack:
    # AutoAttack (AA)
    adversary = AutoAttack(model, norm = 'L2', eps = eps)

    l = [batch['input']  for batch in data.test_batches]
    x_test = torch.cat(l, 0)
    l = [batch['target'] for batch in data.test_batches]
    y_test = torch.cat(l, 0)

    with torch.no_grad():
        adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs = 200)
