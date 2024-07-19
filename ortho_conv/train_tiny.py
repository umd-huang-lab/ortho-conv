import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import argparse
import os
import time
import utils

import numpy as np
from apex import amp

from modules import *
from networks.resnet import ResNet, WideResNet
from networks.convnet import ConvNet, WideConvNet
from networks.shufflenet import ShuffleNet, WideShuffleNet

from networks.resnet_ import WideResNet as WideResNetN

parser = argparse.ArgumentParser()

parser.add_argument('--model', default = 'WideResNet',
                    choices = ['ResNet', 'WideResNet', 'WideResNetN',
                    'ShuffleNet', 'WideShuffleNet', 'ConvNet', 'WideConvNet'])
parser.add_argument('--layers', default = 10, type = int)
parser.add_argument('--factor', default =  1, type = int)
parser.add_argument('--downsampling', default = 'pool',
                    choices = ['pool', 'stride_wide', 'stride_slim'])

parser.add_argument('--kernel', default =  3, type = int)
parser.add_argument('--dilate', default =  1, type = int)
parser.add_argument('--init', default = 'permutation', type = str)

parser.add_argument('--linear', default = 'CayleyLinear',
                    choices = ['nn.Linear', 'BjorckLinear', 'CayleyLinear', 'LieExpLinear'])
parser.add_argument('--conv',   default = 'CayleyConv',
                    choices = ['PlainConv', 'BCOP', 'CayleyConv', 'LieExpConv',
                    'Paraunitary', 'SCFac', 'RKO',  'OSSN'])

parser.add_argument('--norm', default = 'None', choices = ['None', 'nn.BatchNorm2d'])
parser.add_argument('--actv', default = 'GroupSort', choices = ['GroupSort', 'nn.ReLU'])

parser.add_argument('--dataset', default = 'tiny-imagenet', )
parser.add_argument('--data_path', default = './data', type = str)
parser.add_argument('--ckpt_path', default = './checkpoints', type = str)
parser.add_argument('--name', default = 'default', type = str)

parser.add_argument('--epochs', default = 200, type = int)
parser.add_argument('--save_epochs', default = 10, type = int)

parser.add_argument('--batch_size', default = 50, type = int)
parser.add_argument('--lr_max', default = 0.01, type = float)

parser.add_argument('--resume', action = 'store_true')
parser.add_argument('--ckpt_name', default = 'default', type = str)

parser.add_argument('--ce_loss', action = 'store_true')
parser.add_argument('--stddev', action = 'store_true')

parser.add_argument('--eps_0', default = 0.5, type = float) # for training
parser.add_argument('--eps',  default = 36.0, type = float) # for evaluation

parser.add_argument('--distributed', action = 'store_true')
parser.add_argument('--fused_adam', action = 'store_true')
parser.add_argument('--apex', action = 'store_true')
parser.add_argument('--amp', action = 'store_true')

parser.add_argument('--local_rank', default = 0, type = int)

args = parser.parse_args()

# utility for synchronization
def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op = torch.distributed.ReduceOp.SUM)
    return rt

# enable distributed computing
if args.distributed:
    num_devices = torch.cuda.device_count()
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend = 'nccl', init_method = 'env://')

    world_size  = torch.distributed.get_world_size() #os.environ['WORLD_SIZE']
    print('num_devices', num_devices, 'local_rank', args.local_rank, 'world_size', world_size)
else: # if not args.distributed:
    num_devices, world_size = 1, 1

eps = args.eps / 255.0
alpha = eps / 4.0

## model
if args.local_rank == 0:
    print('stddev: ', args.stddev)
    print('linear: ', args.linear)
    print('conv: ',   args.conv)
    print('norm: ',   args.norm)
    print('actv: ',   args.actv)

_model = eval(args.model)
_full = eval(args.linear)
_conv = eval(args.conv)
_norm = eval(args.norm) if args.norm != 'None' else None
_actv = eval(args.actv)

if args.model in ['ResNet', 'ShuffleNet', 'ConvNet']:
    model_name = args.model + str(args.layers)

    model = _model(conv = _conv, linear = _full, 
        in_height = 64, in_width = 64, out_classes = 200,
        num_layers = args.layers, downsampling = args.downsampling,
        conv_kernel = args.kernel, conv_dilate = args.dilate, init = args.init,
    ).cuda()

elif args.model in ['WideResNet', 'WideShuffleNet', 'WideConvNet']:
    model_name = args.model + str(args.layers) + '-' + str(args.factor)

    model = _model(conv = _conv, linear = _full,
        in_height = 64, in_width = 64, out_classes = 200,
        num_layers = args.layers, widen_factor = args.factor, downsampling = args.downsampling,
        conv_kernel = args.kernel, conv_dilate = args.dilate, init = args.init,
    ).cuda()

elif args.model in ['WideResNetN']:
    model_name = args.model + str(args.layers) + '-' + str(args.factor)

    model = _model(conv = _conv, linear = _full, norm = _norm, actv = _actv,
        in_height = 64, in_width = 64, out_classes = 200, 
        num_layers = args.layers, widen_factor = args.factor, downsampling = args.downsampling,
        conv_kernel = args.kernel,
    ).cuda()

if args.name == "default":
    if args.ce_loss:
        name = "%s_%s_ce" % (model_name, args.conv)
    else: # if not args.ce_loss:
        name = "%s_%s_%s" % (model_name, args.conv, args.eps_0)

    if args.local_rank == 0:
        print(name)

## optimizer: Adam
if args.fused_adam:
    from apex.optimizers import FusedAdam
    optimizer = FusedAdam(model.parameters(), lr = args.lr_max)
else: # if not args.use_fused:
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr_max)

if args.amp:
    model, optimizer = amp.initialize(model, optimizer, opt_level = "O1")

if args.distributed:
    if args.apex:
        from apex.parallel import DistributedDataParallel as DDP
        model = DDP(model, delay_allreduce = True)
    else: # if not args.apex
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids = [args.local_rank], find_unused_parameters = True)

## dataset: tiny-ImageNet
if args.local_rank == 0 and not os.path.exists(args.ckpt_path):
    os.mkdir(args.ckpt_path)

ckpt_path = os.path.join(args.ckpt_path, args.dataset)
if args.local_rank == 0 and not os.path.exists(ckpt_path):
    os.mkdir(ckpt_path)

if args.resume:
    ckpt = torch.load(os.path.join(ckpt_path, args.ckpt_name))
    start_epoch = ckpt['epoch'] + 1
    with torch.no_grad():
        model(torch.randn(2, 3, 64, 64).cuda())
        model.load_state_dict(ckpt['state_dict'])
else: # if not args.resume:
    start_epoch = 0

data_path = os.path.join(args.data_path, args.dataset)
if not os.path.exists(args.data_path):
    raise Exception("The dataset does not exist.")

if args.stddev:
    train_transforms = transforms.Compose([
        transforms.RandomCrop(64, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
else: # if not args.stddev:
    train_transforms = transforms.Compose([
        transforms.RandomCrop(64, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

train_data_path = os.path.join(data_path, 'train')
train_dataset = datasets.ImageFolder(train_data_path, train_transforms)

train_samples = len(train_dataset) 
train_sampler = data.distributed.DistributedSampler(train_dataset,
    num_replicas = world_size, rank = args.local_rank, shuffle = True)

train_loader = data.DataLoader(train_dataset,
    batch_size = args.batch_size, drop_last = True, 
    num_workers = num_devices * 4, pin_memory = True, sampler = train_sampler)

if args.stddev:
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
else: # if not args.stddev:
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

test_data_path = os.path.join(data_path, 'test')
test_dataset = datasets.ImageFolder(test_data_path, test_transforms)

test_samples = len(test_dataset)
test_sampler = data.distributed.DistributedSampler(test_dataset,
    num_replicas = world_size, rank = args.local_rank, shuffle = False)

test_loader = data.DataLoader(test_dataset,
    batch_size = args.batch_size, drop_last = True,
    num_workers = num_devices * 4, pin_memory = True, sampler = test_sampler)

# lr schedule: super-convergence
lr_schedule = lambda t: np.interp([t], 
    [0, args.epochs * 2 // 5, args.epochs * 4 // 5, args.epochs], [0, args.lr_max, args.lr_max / 20.0, 0])[0]

# loss: multi-margin loss
if args.ce_loss:
    criterion = lambda outputs, targets: F.cross_entropy(outputs, targets)
else: # if not args.ce_loss:
    criterion = lambda outputs, targets: utils.margin_loss(outputs, targets, args.eps_0, 1.0, 1.0)

for epoch in range(start_epoch, args.epochs):

    # 1) training
    model.train()
    train_acc = torch.tensor(0.).cuda()

    start = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        lr = lr_schedule(epoch + (i + 1) / len(train_loader))
        optimizer.param_groups[0].update(lr = lr)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data) / world_size
        else: # if not distributed:
            reduced_loss = loss

        optimizer.zero_grad()

        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else: # if not args.amp:
            loss.backward()

        optimizer.step()

        train_acc += (outputs.max(1)[1] == targets).sum()

    if args.distributed:
        train_acc = reduce_tensor(train_acc.data).item() / train_samples
    else: # if not args.distributed:
        train_acc = train_acc.item() / train_samples

    if args.local_rank == 0:
        print(f'[{args.model}] Epoch: {epoch} | Train Acc: {train_acc :.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}')

    # 2) testing
    model.eval()
    test_acc = torch.tensor(0.).cuda()

    for inputs, targets in test_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        test_acc += (outputs.max(1)[1] == targets).sum()

    if args.distributed:
        test_acc = reduce_tensor(test_acc.data).item() / test_samples
    else: # if not args.distributed:
        test_acc = test_acc.item() / test_samples

    if args.local_rank == 0:
        print(f'[{args.model}] Epoch: {epoch} | Test Acc: {test_acc :.4f}')

        if epoch % args.save_epochs == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict()
            }, os.path.join(ckpt_path, "%s_%d.pt" % (name, epoch)))

torch.save({
    'epoch': args.epochs - 1,
    'state_dict': model.state_dict()
}, os.path.join(ckpt_path, "%s_%d_last.pt" % (name, epoch)))

# 3) evaluation of robustness
if not args.stddev:
    cert_right, cert_wrong, insc_right, insc_wrong = utils.cert_stats_(model, test_loader, eps * 2**0.5)
    if args.distributed:
        cert_right = reduce_tensor(cert_right.data).item() / test_samples
        cert_wrong = reduce_tensor(cert_wrong.data).item() / test_samples
        insc_right = reduce_tensor(insc_right.data).item() / test_samples
        insc_wrong = reduce_tensor(insc_wrong.data).item() / test_samples
    else: # if args.distributed:
        cert_right = cert_right.item() / test_samples
        cert_wrong = cert_wrong.item() / test_samples
        insc_right = insc_right.item() / test_samples
        insc_wrong = insc_wrong.item() / test_samples

    print('[{}] (PROVABLE) Certifiably Robust (eps: {:.4f}): {:.4f}, Cert. Wrong: {:.4f}, Insc. Right: {:.4f}, Insc. Wrong: {:.4f}'.format(args.model, eps, cert_right, cert_wrong, insc_right, insc_wrong))

robust_acc = utils.rob_acc_(model, test_loader, eps, alpha, optimizer, args.amp, 10, 1, linf_proj = False, l2_grad_update = True)
if args.distributed:
    robust_acc = reduce_tensor(robust_acc.data).item() / test_samples
else: # if not args.distributed:
    robust_acc = robust_acc.item() / test_samples

print('[{}] (EMPIRICAL) Robust accuracy (eps: {:.4f}): {}'.format(args.model, eps, robust_acc))
