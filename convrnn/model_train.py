# system modules
import os, sys, argparse, time
sys.path.insert(0, "./utils")

# basic pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# computer vision/image processing modules 
import torchvision
import skimage.metrics

# math/probability modules
import random
import numpy as np

# custom utilities
from tensorboardX import SummaryWriter
from dataloader import KTH_Dataset, MNIST_Dataset

from utils.network import ConvLSTMNet 
from utils.gpu_affinity import set_affinity
from apex import amp

torch.backends.cudnn.benchmark = True

def main(args):
    ## Distributed computing

    # utility for synchronization
    def reduce_tensor(tensor, reduce_sum = False):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op = torch.distributed.ReduceOp.SUM)
        return rt if reduce_sum else (rt / world_size)

    # enable distributed computing
    if args.distributed:
        set_affinity(args.local_rank)
        num_devices = torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend = 'nccl', init_method = 'env://')

        if 'NGC_MULTINODE_RANK' in os.environ: ## cluster 434
            node_rank  = int(os.environ['NGC_MULTINODE_RANK'])
            print(' -- init cluster 434:', 
                'node_rank', node_rank, 'local_rank', args.local_rank)
        elif 'NGC_ARRAY_INDEX' in os.environ: ## NGC cluster 
            node_rank  = int(os.environ['NGC_ARRAY_INDEX'])
            print(' -- init NGC: ', 
                'node_rank', node_rank, 'local_rank', args.local_rank)
        elif 'CLUSTER' in os.environ and os.environ['CLUSTER'] == 'DRACO': ## Draco
            world_size = int(os.environ['WORLD_SIZE'])
            node_rank  = int(os.environ['SLURM_NODEID'])
            local_rank = int(os.environ['LOCAL_RANK'])
            print(' -- init Draco:', 'world_size', world_size, 
                'node_rank', node_rank, 'local_rank', args.local_rank)
        else: 
            node_rank = args.node_rank
            print('-- No multinode')

        global_rank = node_rank * num_devices + args.local_rank
        world_size  = torch.distributed.get_world_size() #os.environ['WORLD_SIZE']
        print('node_rank', node_rank, 
            'num_devices', num_devices,
             'local_rank', args.local_rank,
            'global_rank', global_rank, 
            'world_size', world_size)
    else:
        global_rank, num_devices, world_size = 0, 1, 1


    ## Model preparation (Conv-LSTM or Conv-TT-LSTM)

    # size of the neural network model (depth and width)
    if args.model_size == "origin": # 12-layers
        layers_per_block = (3, 3, 3, 3)
        hidden_channels  = (32, 48, 48, 32)
        skip_stride = 2
    elif args.model_size == "shallow8": # 8-layers
        layers_per_block = (2, 2, 2, 2)
        hidden_channels  = (32, 48, 48, 32)
        skip_stride = 2
    elif args.model_size == "shallow4": # 8-layers
        layers_per_block = (1, 1, 1, 1)
        hidden_channels  = (32, 48, 48, 32)
        skip_stride = 2
    elif args.model_size == "test":  # 4-layers
        layers_per_block = (4, )
        hidden_channels  = (128, )
        skip_stride = None 
    else:
        raise NotImplementedError

    # construct the model with the specified hyper-parameters
    model = ConvLSTMNet(
        input_channels = args.img_channels, 
        output_sigmoid = args.use_sigmoid,
        # model architecture
        layers_per_block = layers_per_block, 
        hidden_channels  = hidden_channels, 
        skip_stride = skip_stride,
        # convolutional tensor-train layers
        cell = args.model, 
        cell_params = {
            "order": args.model_order,
            "steps": args.model_steps,
            "ranks": args.model_ranks,
            "ortho_states": args.ortho_states,
            "ortho_inputs": args.ortho_inputs,
            "ortho_init": args.ortho_init,
            "activation": args.activation,
            "gain": args.init_gain,
            "norm": args.use_norm,
        },
        # convolutional parameters
        kernel_size = args.kernel_size, 
        bias = args.use_bias
    ).cuda()

    model_name = args.model_name + '_' + args.model_stamp
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    if args.local_rank == 0:
        print("Model name: %s (Params: %d)"% (model_name, num_params))
        if args.model == "convlstm": 
            print("Model type: %s" % args.model)
        elif args.model == "convttlstm":
            print("Model type: %s (order: %d, steps: %d, ranks: %d)" 
                % (args.model, args.model_order, args.model_steps, args.model_ranks))
        elif args.model == "ttconvlstm":
            print("Model type: %s (order: %d, ranks: %d)" 
                % (args.model, args.model_order, args.model_ranks))
        elif args.model in ["convrnn", "convgru"]:
            print("Model type: %s (ortho_states: %s, ortho_inputs: %s, ortho_init: %s)"
                % (args.model, args.ortho_states, args.ortho_inputs, args.ortho_init))
            print("activation: %s, norm: %s, gain: %d"
                % args.activation, args.use_norm, args.init_gain)

    ## Dataset Preparation (KTH, MNIST)
    total_batch_size  = args.batch_size
    assert total_batch_size % world_size == 0, \
        'The batch_size is not divisible by world_size.'
    batch_size = total_batch_size // world_size

    total_valid_batch_size = args.valid_batch_size
    assert total_valid_batch_size % world_size == 0, \
        'The valid batch size is not divisible by world_size.'
    valid_batch_size = total_valid_batch_size // world_size

    assert args.dataset in ["MNIST", "MNIST100", "KTH"], \
        "The dataset is not currently supported."

    Dataset = {"KTH": KTH_Dataset, "MNIST": MNIST_Dataset, "MNIST100": MNIST_Dataset}[args.dataset]

    # path to the dataset folder
    if args.data_path == "default":
        datafolders = {"KTH": "kth", "MNIST": "moving-mnist", "MNIST100": "moving-mnist-f100"} 
        DATA_DIR = os.path.join(args.data_base, datafolders[args.dataset])
    else: # if args.data_path != "default":
        DATA_DIR = args.data_path

    assert os.path.exists(DATA_DIR), \
        "The dataset folder does not exist. "+DATA_DIR


    # dataloader for the training dataset
    train_data_path = os.path.join(DATA_DIR, args.train_data_file)
    assert os.path.exists(train_data_path), \
        "The training dataset does not exist. "+train_data_path

    train_dataset = Dataset({"path": train_data_path, "unique_mode": False,
        "num_frames": args.input_frames + args.future_frames, "num_samples": args.train_samples, 
        "height": args.img_height, "width": args.img_width, "channels": args.img_channels})

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas = world_size, rank = global_rank, shuffle = True)

    train_loader  = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, drop_last = True,
        num_workers = num_devices * args.worker, pin_memory = True, sampler = train_sampler)

    train_samples = len(train_loader) * total_batch_size

    # dataloaer for the valiation dataset 
    valid_data_path = os.path.join(DATA_DIR, args.valid_data_file)
    assert os.path.exists(valid_data_path), \
        "The validation dataset does not exist."

    valid_dataset = Dataset({"path": valid_data_path, "unique_mode": True,
        "num_frames": args.input_frames + args.future_frames, "num_samples": args.valid_samples,
        "height": args.img_height, "width": args.img_width, "channels": args.img_channels})

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas = world_size, rank = global_rank, shuffle = False)
    valid_loader  = torch.utils.data.DataLoader(
        valid_dataset, batch_size = valid_batch_size, drop_last = True, 
        num_workers = num_devices * args.worker, pin_memory = True, sampler = valid_sampler)

    valid_samples = len(valid_loader) * total_valid_batch_size
    
    if args.local_rank == 0:
        print("Training samples: %s/%s; Validation samples: %s/%s" 
            % (train_samples, train_dataset.data_samples, 
               valid_samples, valid_dataset.data_samples))
    

    ## Outputs (Models and Results)
    if args.output_path == "default":
        output_folders = {"KTH": "kth", "MNIST": "moving-mnist"}
        OUTPUT_DIR = os.path.join(args.output_base, output_folders[args.dataset])
    else: # if args.output_path != "default":
        OUTPUT_DIR = args.output_path

    OUTPUT_DIR = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.exists(OUTPUT_DIR) and global_rank == 0:
        os.makedirs(OUTPUT_DIR)
      
    # path to the models folder
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    if not os.path.exists(MODEL_DIR) and global_rank == 0:
        os.makedirs(MODEL_DIR)

    # path to the results folder
    RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
    if not os.path.exists(RESULT_DIR) and global_rank == 0:
        os.makedirs(RESULT_DIR)

    # path to the validation images folder
    RESULT_IMG = os.path.join(RESULT_DIR, "valid_images")
    if not os.path.exists(RESULT_IMG) and global_rank == 0:
        os.makedirs(RESULT_IMG)

    # path to the tensorboard folder 
    RESULT_TBW = os.path.join(RESULT_DIR, "tensorboardX")
    if global_rank == 0:
        if not os.path.exists(RESULT_TBW):
            os.makedirs(RESULT_TBW)
        tensorboard_writer = SummaryWriter(RESULT_TBW)


    ## Hyperparameters for learning algorithm
    if not args.start_begin:
    # if os.path.exists(MODEL_DIR):
        if args.start_last:
            if args.start_best:
                MODEL_FILE = os.path.join(MODEL_DIR, 'training_best.pt')
            else: # (--start-last)
                MODEL_FILE = os.path.join(MODEL_DIR, 'training_last.pt')
        else: # (--start_spec)
                MODEL_FILE = os.path.join(MODEL_DIR, "training_%d.pt" % args.start_epoch)

        assert os.path.exists(MODEL_FILE), "The model is not found in the folder."
    else: # (--start-begin)
        MODEL_FILE = None

    if MODEL_FILE is not None:
        checkpoint = torch.load(MODEL_FILE)

        # recover the information of training progress
        start_epoch = checkpoint.get("epoch", args.start_epoch)
        total_samples = checkpoint.get("total_samples", start_epoch * train_samples)

        # recover the epoch/loss of the best model
        min_epoch = checkpoint.get("min_epoch", 0)
        min_loss  = checkpoint.get("min_loss", float("inf"))

        max_epoch = checkpoint.get("max_epoch", 0)
        max_ssim  = checkpoint.get("max_ssim",  0)

        # recover the scheduled sampling ratio
        teacher_forcing = checkpoint.get("teacher_forcing", args.teacher_forcing)
        scheduled_sampling_ratio = checkpoint.get("scheduled_sampling_ratio", 1)
        ssr_decay_start = checkpoint.get("ssr_decay_start", args.ssr_decay_start) 
        ssr_decay_mode  = checkpoint.get("ssr_decay_mode", False)
        
        # recover the learning rate
        lr_decay_start = checkpoint.get("lr_decay_start", 
            args.num_epochs if teacher_forcing else args.lr_decay_start)
        learning_rate  = checkpoint.get("learning_rate", args.learning_rate)
        lr_decay_mode  = checkpoint.get("lr_decay_mode", False)

    else: # if MODEL_FILE is None:
        start_epoch, total_samples = 0, 0
        min_epoch, min_loss = 0, float("inf")
        max_epoch, max_ssim = 0, 0

        # intialize the scheduled sampling ratio
        teacher_forcing = args.teacher_forcing
        scheduled_sampling_ratio = 1
        ssr_decay_start = args.ssr_decay_start
        ssr_decay_mode  = False

        # initialize the learning rate
        learning_rate  = args.learning_rate
        lr_decay_start = args.num_epochs if teacher_forcing else args.lr_decay_start
        lr_decay_mode  = False

    # loss function for training
    if args.loss_function == "l1": 
        loss_func = lambda outputs, targets: \
            F.l1_loss(outputs, targets, reduction = "mean")
    elif args.loss_function == "l2": 
        loss_func = lambda outputs, targets: \
            F.mse_loss(outputs, targets, reduction = "mean")
    elif args.loss_function == "l1l2":
        loss_func = lambda outputs, targets: \
             F.l1_loss(outputs, targets, reduction = "mean") + \
            F.mse_loss(outputs, targets, reduction = "mean")
    else:
        raise NotImplementedError 


    ## Main script for training and validation
    if args.use_fused:
        from apex.optimizers import FusedAdam
        optimizer = FusedAdam(model.parameters(), lr = learning_rate)
    else: # if not args.use_fused:
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level = "O1")

    if args.distributed:
        if args.use_apex: # use DDP from apex.parallel
            from apex.parallel import DistributedDataParallel as DDP
            model = DDP(model, delay_allreduce = True)
        else: # use DDP from nn.parallel
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids = [args.local_rank])

    if MODEL_FILE is not None:
        # recover the model parameters (weights)
        model.load_state_dict(checkpoint["model_state_dict"])

        if args.use_amp:
            amp.load_state_dict(checkpoint["amp_state_dict"])

    for epoch in range(start_epoch, args.num_epochs):
        if global_rank == 0:
            # log the hyperparameters of learning sheduling 
            tensorboard_writer.add_scalar('Train/lr',  optimizer.param_groups[0]['lr'], epoch + 1)
            tensorboard_writer.add_scalar('Train/ssr', scheduled_sampling_ratio, epoch + 1)

        ## Phase 1: Learning on the training set
        model.train()
        samples, LOSS = 0, 0.

        start_time = time.time()

        for it, frames in enumerate(train_loader):

            samples += total_batch_size
            total_samples += total_batch_size

            if args.img_channels == 1:
                frames = torch.mean(frames, dim = -1, keepdim = True)

            # 5-th order: batch_size(0) x total_frames(1) x channels(2) x height(3) x width(4) 
            frames = frames.permute(0, 1, 4, 2, 3).cuda()

            inputs = frames[:, :-1] if teacher_forcing else frames[:, :args.input_frames] 
            origin = frames[:, -args.output_frames:]

            pred = model(inputs, 
                input_frames  =  args.input_frames, 
                future_frames = args.future_frames, 
                output_frames = args.output_frames, 
                teacher_forcing = teacher_forcing, 
                scheduled_sampling_ratio = scheduled_sampling_ratio, 
                checkpointing = args.use_checkpointing)

            # compute the loss function
            loss = loss_func(pred, origin)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data

            LOSS += reduced_loss.item() * total_batch_size

            # compute the backpropagation
            optimizer.zero_grad()
            
            # gradient clipping and stochastic gradient descent
            if args.use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if args.gradient_clipping:
                    grad_norm = nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.clipping_threshold)
            else: # if not args.use_amp:
                loss.backward()
                if args.gradient_clipping:
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), args.clipping_threshold)

            optimizer.step()

            # compute the gradient norm (after clipping)
            if (it + 1) % args.log_iterations == 0: 
                LOSS /= args.log_iterations * total_batch_size

                if args.local_rank == 0:
                    if args.gradient_clipping:
                        if global_rank == 0:
                            tensorboard_writer.add_scalar("Gradient_norm/grad-clip", grad_norm, total_samples)
                        print('-- grad-clip: {}'.format(grad_norm))

                    print('Epoch: {}/{}, Training: {}/{}, Loss: {}'.format(
                        epoch + 1, args.num_epochs, samples, train_samples, LOSS))
                    if global_rank == 0:
                        tensorboard_writer.add_scalar('Train/LOSS', LOSS, total_samples)

                LOSS = 0.

        print("time", time.time() - start_time)

        ## Phase 2: Evaluation on the validation set
        model.eval()
        origin_norm = 0.
        samples, LOSS = 0, 0.

        MSE_  = torch.zeros((args.future_frames), dtype = torch.float32).cuda()
        PSNR_ = torch.zeros((args.future_frames), dtype = torch.float32).cuda()
        SSIM_ = torch.zeros((args.future_frames), dtype = torch.float32).cuda()

        with torch.no_grad():
            for it, frames in enumerate(valid_loader):
                samples += total_valid_batch_size

                if args.img_channels == 1:
                    frames = torch.mean(frames, dim = -1, keepdim = True)

                # 5-th order: batch_size x total_frames x channels x height x width 
                frames = frames.permute(0, 1, 4, 2, 3).cuda()

                inputs = frames[:,  :args.input_frames]
                origin = frames[:, -args.output_frames:]

                pred = model(inputs, 
                    input_frames  =  args.input_frames, 
                    future_frames = args.future_frames, 
                    output_frames = args.output_frames, 
                    teacher_forcing = False, 
                    checkpointing   = False)

                loss =  loss_func(pred, origin)

                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data)
                else:
                    reduced_loss = loss.data

                LOSS += reduced_loss.item() * total_valid_batch_size

                # clamp the output to [0, 1]
                pred = torch.clamp(pred, min = 0, max = 1)

                # save the first sample for each batch to the tensorboard
                if global_rank == 0 and it % args.log_iterations == 0:
                    input_0 = inputs[0, -args.future_frames:] if args.input_frames >= args.future_frames \
                        else torch.cat([torch.zeros(args.future_frames - args.input_frames, 
                            args.img_channels, args.img_height, args.img_width, device = device), inputs[0]], dim = 0)

                    origin_0 = origin[0, -args.future_frames:]
                    pred_0   = pred[0,   -args.future_frames:]

                    img = torchvision.utils.make_grid(torch.cat(
                        [input_0, origin_0, pred_0], dim = 0), nrow = args.future_frames)

                    tensorboard_writer.add_image("img_results", img, epoch + 1)

                    RESULT_FILE = os.path.join(RESULT_IMG, "cmp_%d_%d.jpg" % (epoch + 1, samples))
                    print(' = save img results at', RESULT_FILE)
                    torchvision.utils.save_image(img, RESULT_FILE)

                # accumlate the statistics
                origin = origin.permute(0, 1, 3, 4, 2).cpu().numpy()
                pred   =   pred.permute(0, 1, 3, 4, 2).cpu().numpy()
                
                for i in range(valid_batch_size):
                    for t in range(-args.future_frames, 0):
                        origin_, pred_ = origin[i, t], pred[i, t]
                        if args.img_channels == 1:
                            origin_ = np.squeeze(origin_, axis = -1)
                            pred_   = np.squeeze(pred_,   axis = -1)

                        MSE_[t]  += skimage.metrics.mean_squared_error(origin_, pred_)
                        PSNR_[t] += skimage.metrics.peak_signal_noise_ratio(origin_, pred_)
                        SSIM_[t] += skimage.metrics.structural_similarity(origin_, pred_, multichannel = args.img_channels > 1)

        if args.distributed:
            MSE  = reduce_tensor(MSE_,  reduce_sum = True) / valid_samples
            PSNR = reduce_tensor(PSNR_, reduce_sum = True) / valid_samples
            SSIM = reduce_tensor(SSIM_, reduce_sum = True) / valid_samples
        else:
            MSE  = MSE_  / valid_samples
            PSNR = PSNR_ / valid_samples
            SSIM = SSIM_ / valid_samples

        LOSS_AVG = LOSS / valid_samples
        MSE_AVG  = torch.mean(MSE ).cpu().item()
        PSNR_AVG = torch.mean(PSNR).cpu().item()
        SSIM_AVG = torch.mean(SSIM).cpu().item()

        if args.local_rank == 0:
            print("Epoch {}, LOSS: {}, MSE: {} (x1e-3); PSNR: {}, SSIM: {}".format(
                epoch + 1, LOSS_AVG, 1e3 * MSE_AVG, PSNR_AVG, SSIM_AVG))

            if global_rank == 0:
                tensorboard_writer.add_scalar("Val/LOSS", LOSS, epoch + 1)
                tensorboard_writer.add_scalar("Val/MSE",  MSE_AVG,  epoch + 1)
                tensorboard_writer.add_scalar("Val/PSNR", PSNR_AVG, epoch + 1)
                tensorboard_writer.add_scalar("Val/SSIM", SSIM_AVG, epoch + 1)

        # automatic scheduling of (1) learning rate and (2) scheduled sampling ratio
        if LOSS_AVG < min_loss:
            min_epoch, min_loss = epoch + 1, LOSS_AVG

        if SSIM_AVG > max_ssim:
            max_epoch, max_ssim = epoch + 1, SSIM_AVG

        if not ssr_decay_mode and epoch > ssr_decay_start and epoch > min_epoch + args.decay_log_epochs:
            ssr_decay_mode = True
            lr_decay_start = epoch + args.lr_decay_start 

        if not lr_decay_mode and epoch > lr_decay_start and epoch > min_epoch + args.decay_log_epochs:
            lr_decay_mode = True

        if ssr_decay_mode and (epoch + 1) % args.ssr_decay_epoch == 0:
            scheduled_sampling_ratio = max(scheduled_sampling_ratio - args.ssr_decay_ratio, 0)

        if lr_decay_mode and (epoch + 1) % args.lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_rate


        ## Saving the checkpoint
        if global_rank == 0:
            checkpoint = {
                'epoch': epoch + 1, 'total_samples': total_samples, # training progress 
                'min_epoch': min_epoch, 'min_loss': min_loss, # best model and loss
                'max_epoch': max_epoch, 'max_ssim': max_ssim, # best model and ssim
                'model_state_dict': model.state_dict(), # model parameters
                # scheduled sampling ratio
                'teacher_forcing': teacher_forcing,
                'scheduled_sampling_ratio': scheduled_sampling_ratio,
                'ssr_decay_start': ssr_decay_start, 'ssr_decay_mode': ssr_decay_mode,
                # learning rate of the optimizer
                'learning_rate': optimizer.param_groups[0]['lr'],
                'lr_decay_start':  lr_decay_start,  'lr_decay_mode':  lr_decay_mode}

            if args.use_amp:
                checkpoint["amp_state_dict"] = amp.state_dict()

            torch.save(checkpoint, os.path.join(MODEL_DIR, 'training_last.pt'))

            if (epoch + 1) % args.save_epoch == 0:
                torch.save(checkpoint, os.path.join(MODEL_DIR, 'training_%d.pt' % (epoch + 1)))

            if (epoch + 1) == min_epoch:
                print(' = save model: ',    os.path.join(MODEL_DIR, 'training_best.pt'))
                torch.save(checkpoint, os.path.join(MODEL_DIR, 'training_best.pt'))

            if (epoch + 1) == max_epoch:
                print(' = save model: ',    os.path.join(MODEL_DIR, 'training_best_ssim.pt'))
                torch.save(checkpoint, os.path.join(MODEL_DIR, 'training_best_ssim.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Conv-TT-LSTM Training")

    ## Devices (Single GPU / Distributed computing)

    # whether to use distributed computing
    parser.add_argument('--use-distributed', dest = "distributed", 
        action = 'store_true',  help = 'Use distributed computing in training.')
    parser.add_argument('--no-distributed',  dest = "distributed", 
        action = 'store_false', help = 'Use single process (GPU) in training.')
    parser.set_defaults(distributed = True)

    parser.add_argument('--use-fused', dest = 'use_fused', 
        action = 'store_true',  help = 'Use fused kernels in training.')
    parser.add_argument( '--no-fused', dest = 'use_fused',
        action = 'store_false', help =  'No fused kernels in training.')
    parser.set_defaults(use_fused = False)

    parser.add_argument('--use-apex', dest = 'use_apex', 
        action = 'store_true',  help = 'Use DistributedDataParallel from apex.')
    parser.add_argument( '--no-apex', dest = 'use_apex', 
        action = 'store_false', help = 'Use DistributedDataParallel from nn.distributed.')
    parser.set_defaults(use_apex = False)

    parser.add_argument('--use-amp', dest = 'use_amp', 
        action = 'store_true',  help = 'Use automatic mixed precision in training.')
    parser.add_argument( '--no-amp', dest = 'use_amp', 
        action = 'store_false', help = 'No automatic mixed precision in training.')
    parser.set_defaults(use_amp = False)

    parser.add_argument('--use-checkpointing', dest = 'use_checkpointing', 
        action = 'store_true',  help = 'Use checkpointing to reduce memory utilization.')
    parser.add_argument( '--no-checkpointing', dest = 'use_checkpointing', 
        action = 'store_false', help = 'No checkpointing (faster training).')
    parser.set_defaults(use_checkpointing = False)

    parser.add_argument('--worker', default = 4, type = int, 
        help = 'number of workers for DataLoader.')

    parser.add_argument('--local_rank', default = 0, type = int)
    parser.add_argument( '--node_rank', default = 0, type = int)

    ## Data format (batch x steps x height x width x channels)

    # batch size (0) 
    parser.add_argument('--batch-size', default = 16, type = int,
        help = 'The total batch size in each training iteration.')
    parser.add_argument('--valid-batch-size', default = 16, type = int,
        help = 'The total batch size in each validation iteration.')
    parser.add_argument('--log-iterations', default = 10, type = int,
        help = 'Log the statistics every log_iterations.')

    # frame split (1)
    parser.add_argument('--input-frames',  default = 10, type = int,
        help = 'The number of input frames to the model.')
    parser.add_argument('--future-frames', default = 10, type = int,
        help = 'The number of predicted frames of the model.')
    parser.add_argument('--output-frames', default = 19, type = int,
        help = 'The number of output frames of the model.')

    # frame format (2, 3, 4)
    parser.add_argument('--img-height',  default = 120, type = int, 
        help = 'The image height of each video frame.')
    parser.add_argument('--img-width',   default = 120, type = int, 
        help = 'The image width of each video frame.')
    parser.add_argument('--img-channels', default =  3, type = int, 
        help = 'The number of channels in each video frame.')

    ## Models (Conv-LSTM or Conv-TT-LSTM)

    # model name (with time stamp as suffix)
    parser.add_argument('--model-name', default = "test", type = str,
        help = 'The model name is used to create the folder names.')
    parser.add_argument('--model-stamp', default = "0000", type = str, 
        help = 'The stamp is used to create the suffix to the model name.')

    # model type and size (depth and width) 
    parser.add_argument('--model', default = 'convlstm', type = str,
        help = 'The model is \"convlstm\", \"convttlstm\", \"ttconvlstm\", \"convrnn\", or \"convgru\".')
    parser.add_argument('--model-size', default = 'origin', type = str,
        help = 'The model size is \"origin\", \"shallow8\", \"shallow4\", or \"test\"')
    
    parser.add_argument('--use-sigmoid', dest = 'use_sigmoid', 
        action = 'store_true',  help = 'Use sigmoid function at the output of the model.')
    parser.add_argument('--no-sigmoid',  dest = 'use_sigmoid', 
        action = 'store_false', help = 'Use output from the last layer as the final output.')
    parser.set_defaults(use_sigmoid = False)

    # parameters of the convolutional tensor-train layers
    parser.add_argument('--model-order', default = 3, type = int, 
        help = 'The order of the convolutional tensor-train LSTMs.')
    parser.add_argument('--model-steps', default = 3, type = int, 
        help = 'The steps of the convolutional tensor-train LSTMs')
    parser.add_argument('--model-ranks', default = 8, type = int, 
        help = 'The tensor rank of the convolutional tensor-train LSTMs.')

    # parameters of the orthogonal convolutional recurrent layers (model: ConvRNNCell)
    parser.add_argument('--use-ortho-states', dest = "ortho_states", action = 'store_true',
        help = "Use orthogonal convolution in the hidden-hidden transition.")
    parser.add_argument( '--no-ortho-states', dest = "ortho_states", action = 'store_false',
        help = "Use standard convolution in the hidden-hidden transition.")
    parser.set_defaults(ortho_states = True)

    parser.add_argument('--use-ortho-inputs', dest = "ortho_inputs", action = 'store_true',
        help = "Use orthogonal convolution in the input-hidden transition.")
    parser.add_argument('--no-ortho-inputs', dest = "ortho_inputs", action = 'store_false',
        help = "Use standard convolution in the input-hidden transition.")
    parser.set_defaults(ortho_inputs = False)

    parser.add_argument('--ortho-init', default = "torus", type = str, 
        help = "The initialization method for the orthogonal convolutional kernels.")

    parser.add_argument('--activation', default = "modrelu", type = str,
        help = "The activation function for the recurrent layer.")

    # parameters of the convolutional operations
    parser.add_argument('--kernel-size', default = 5, type = int, 
        help = "The kernel size of the convolutional operations.")

    parser.add_argument('--init-gain', default = 1, type = float, 
        help = "The intialization gain of the standard convolution")

    parser.add_argument('--use-norm', dest = "use_norm", action = 'store_true',
        help = "Add normalization after the standard convolution.")
    parser.add_argument( '--no-norm', dest = "use_norm", action = 'store_false',
        help = "Do not use normalization after the standard convolution.")
    parser.set_defaults(use_norm = False)

    parser.add_argument('--use-bias', dest = "use_bias", action = 'store_true',
        help = "Add bias term after each convoltional operation.")
    parser.add_argument('--no-bias',  dest = "use_bias", action = 'store_false',
        help = "Do not add bias after the convolutional operations.")
    parser.set_defaults(use_bias = True)

    ## Dataset (Input to the training algorithm)
    parser.add_argument('--dataset', default = "MNIST", type = str, 
        help = 'The dataset name. (Options: KTH, MNIST)')

    parser.add_argument('--data-base', default = '../data/', type = str, 
        help = "The base path to the datasets.")
    parser.add_argument('--data-path', default = 'default', type = str,
        help = 'The path to the dataset folder.')

    # training dataset
    parser.add_argument('--train-data-file', default = 'train', type = str,
        help = 'Name of the folder/file for training set.')
    parser.add_argument('--train-samples', default = 10000, type = int,
        help = 'Number of samples in each training epoch.')

    # validation dataset
    parser.add_argument('--valid-data-file', default = 'valid', type = str, 
        help = 'Name of the folder/file for validation set.')
    parser.add_argument('--valid-samples', default = 5000, type = int, 
        help = 'Number of unique samples in validation set.')

    ## Results and Models (Output from the training algorithm)
    parser.add_argument('--output-base', default = './checkpoints/', type = str, 
        help = "The base path to the outputs.")
    parser.add_argument('--output-path', default = 'default', type = str,
        help = "The path to the folder storing the outputs (models and results).")

    ## Learning algorithm

    # loss function for training
    parser.add_argument('--loss-function', default = 'l1l2', type = str, 
        help = 'The loss function for training.')
    parser.add_argument('--perceptual-loss', dest = "perceptual_loss", action = 'store_true',
        help = "Add perceptual loss for training.")

    # total number of epochs and the interval to save a checkpoint
    parser.add_argument('--num-epochs', default = 800, type = int, 
        help = 'Number of total epochs in training.')
    parser.add_argument('--save-epoch', default =  2, type = int, 
        help = 'Save the model parameters every save_epoch.')

    # the epoch to start/resume training
    parser.add_argument('--start-begin', dest = 'start_begin', action = 'store_true', 
        help = 'Start training a adam-lr0,001-explr-gamma0.96-gs model from the beginning.')
    parser.add_argument('--start-exist', dest = 'start_begin', action = 'store_false',
        help = 'Resume training an existing model.')
    parser.set_defaults(start_begin = True)

    parser.add_argument('--start-last', dest = 'start_last', action = 'store_true', 
    help = 'Resume training from the last available model')
    parser.add_argument('--start-spec', dest = 'start_last', action = 'store_false', 
        help = 'Resume training from the model of the specified epoch')
    parser.set_defaults(start_last = True)

    parser.add_argument('--start-best', dest = 'start_best', action = 'store_true', 
        help = 'Resume training from the best available model')
    parser.set_defaults(start_best = False)

    parser.add_argument('--start-epoch', default = 0, type = int, 
        help = 'The number of epoch to resume training.')

    # logging for automatic scheduling
    parser.add_argument('--decay-log-epochs', default = 20, type = int, 
        help = 'The window size to determine automatic scheduling.')

    # gradient clipping
    parser.add_argument('--gradient-clipping', dest = 'gradient_clipping', 
        action = 'store_true',  help = 'Use gradient clipping in training.')
    parser.add_argument(       '-no-clipping', dest = 'gradient_clipping', 
        action = 'store_false', help = 'No gradient clipping in training.')
    parser.set_defaults(gradient_clipping = False)

    parser.add_argument('--clipping-threshold', default = 1, type = float,
        help = 'The threshold value for gradient clipping.')

    # learning rate
    parser.add_argument('--learning-rate', default = 1e-3, type = float,
        help = 'Initial learning rate of the Adam optimizer.')
    parser.add_argument('--lr-decay-start', default = 20, type = int,
        help = 'The minimum epoch (after scheduled sampling) to start learning rate decay.')
    parser.add_argument('--lr-decay-epoch', default = 5, type = int,
        help = 'The learning rate is decayed every decay_epoch.')
    parser.add_argument('--lr-decay-rate', default = 0.98, type = float,
        help = 'The learning rate by decayed by decay_rate every epoch.')
    parser.add_argument('--eps', default = 1e-8, type = float,
        help = 'Epsilon of the Adam optimizer.')

    # scheduled sampling ratio
    parser.add_argument('--teacher-forcing', dest = 'teacher_forcing', 
        action = 'store_true',  help = 'Use teacher forcing (with scheduled sampling) in training.')
    parser.add_argument(     '--no-forcing', dest = 'teacher_forcing', 
        action = 'store_false', help = 'Training without teacher forcing (with scheduled sampling).')
    parser.set_defaults(teacher_forcing = True)

    parser.add_argument('--ssr-decay-start', default = 20, type = int,
        help = 'The minimum epoch to start scheduled sampling.')
    parser.add_argument('--ssr-decay-epoch', default =  1, type = int, 
        help = 'Decay the scheduled sampling every ssr_decay_epoch.')
    parser.add_argument('--ssr-decay-ratio', default = 2e-3, type = float,
        help = 'Decay the scheduled sampling by ssr_decay_ratio every time.')

    main(parser.parse_args())
