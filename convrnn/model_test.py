# system modules
import os, sys, argparse
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
from utils.convlstmnet import ConvLSTMNet 
from dataloader import KTH_Dataset, MNIST_Dataset

from utils.gpu_affinity import set_affinity

# perceptive quality
import PerceptualSimilarity.models as PSmodels


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
        

    ## Data format: batch(0) x steps(1) x height(2) x width(3) x channels(4) 

    # batch_size (0)
    total_batch_size  = args.batch_size
    assert total_batch_size % world_size == 0, \
        'The batch_size is not divisible by world_size.'
    batch_size = total_batch_size // world_size

    # steps (1)
    total_frames = args.input_frames + args.future_frames

    list_input_frames  = list(range(0,  args.input_frames, args.log_frames))
    plot_input_frames  = len(list_input_frames)

    list_future_frames = list(range(0, args.future_frames, args.log_frames))
    plot_future_frames = len(list_future_frames)

    # frame format (2, 3)
    img_resize = (args.img_height != args.img_height_u) or (args.img_width != args.img_width_u)


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
            print("activation: %s, norm: %s, gain: %s"
                % args.activation, args.use_norm, args.init_gain)
    if args.distributed:
        if args.use_apex: # use DDP from apex.parallel
            from apex.parallel import DistributedDataParallel as DDP
            model = DDP(model, delay_allreduce = True)
        else: # use DDP from nn.parallel
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids = [args.local_rank])

    PSmodel = PSmodels.PerceptualLoss(model = 'net-lin', 
        net = 'alex', use_gpu = True, gpu_ids = [args.local_rank])


    ## Dataset Preparation (KTH, UCF, tinyUCF)
    assert args.dataset in ["MNIST", "KTH"], \
        "The dataset is not currently supported."

    Dataset = {"KTH": KTH_Dataset, "MNIST": MNIST_Dataset}[args.dataset]
               
    # path to the dataset folder
    if args.data_path == "default":
        datafolders = {"KTH": "datasets/kth", "MNIST": "moving-mnist"} 
        DATA_DIR = os.path.join(args.data_base, datafolders[args.dataset])
    else: # if args.data_path != "default":
        DATA_DIR = args.data_path

    assert os.path.exists(DATA_DIR), \
        "The dataset folder does not exist. "+DATA_DIR

    # dataloaer for the valiation dataset 
    test_data_path = os.path.join(DATA_DIR, args.test_data_file)
    assert os.path.exists(test_data_path), \
        "The test dataset does not exist. "+test_data_path

    test_dataset = Dataset({"path": test_data_path, 
        "unique_mode": True, "num_frames": total_frames, "num_samples": args.test_samples,
        "height": args.img_height, "width": args.img_width, "channels": args.img_channels})

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas = world_size, rank = global_rank, shuffle = False)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset, batch_size = batch_size, drop_last = True, 
        num_workers = num_devices * args.worker, pin_memory = True, sampler = test_sampler)

    test_samples = len(test_loader) * total_batch_size

    if args.local_rank == 0:
        print("Test samples: %s/%s" % (test_samples, test_dataset.data_samples))
    
    ## Models and Results
    if args.output_path == "default":
        output_folders = {"KTH": "kth", "MNIST": "moving-mnist"}
        OUTPUT_DIR = os.path.join(args.output_base, output_folders[args.dataset])
    else: # if args.output_path != "default":
        OUTPUT_DIR = args.output_path

    # path to the outputs
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, args.model_name)
    assert os.path.exists(OUTPUT_DIR), \
        "The outputs folder \"%s\" does not exist: " % OUTPUT_DIR

    # path to the models
    MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")
    assert os.path.exists(MODEL_DIR), \
        "The models folder does not exist."

    # load the best / last / specified model
    if args.eval_auto:
        if args.eval_best:
            if args.eval_best_ssim:
                MODEL_FILE = os.path.join(MODEL_DIR, 'training_best_ssim.pt')
            else: # if args.eval_best_loss:
                MODEL_FILE = os.path.join(MODEL_DIR, 'training_best.pt')
        else: # if args.eval_last:
            MODEL_FILE = os.path.join(MODEL_DIR, 'training_last.pt')
    else: # if args.eval_spec:
        MODEL_FILE = os.path.join(MODEL_DIR, 'training_%d.pt' % args.eval_epoch)

    assert os.path.exists(MODEL_FILE), \
        "The specified model is not found in the folder."

    checkpoint = torch.load(MODEL_FILE)
    eval_epoch = checkpoint.get("epoch", args.eval_epoch)
    model.load_state_dict(checkpoint["model_state_dict"])

    # prints and path to the results (images and statistics)
    if args.local_rank == 0:
        print("Model name: \t%s" % (args.model_name))
        print("Model: %s (Size: %s)" % (args.model, args.model_size), end = " \t")
        if args.model != "convlstm":
            print("TT: \tOrder: %d Steps: %s Rank: %s" % 
                (args.model_order, args.model_steps, args.model_ranks), end = " \t")
        print("\n# of params.: \t", num_params, "\t# of future frames: ", args.future_frames)


    RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
    if not os.path.exists(RESULT_DIR) and global_rank == 0:
        os.makedirs(RESULT_DIR)

    RESULT_IMG  = os.path.join(RESULT_DIR, "test_images_" 
        + str(eval_epoch) + "_" + str(args.future_frames))
    if not os.path.exists(RESULT_IMG) and global_rank == 0:
        os.makedirs(RESULT_IMG)
  
    RESULT_STAT = os.path.join(RESULT_DIR, "test_stats")
    if not os.path.exists(RESULT_STAT) and global_rank == 0:
        os.makedirs(RESULT_STAT)

    RESULT_STAT = os.path.join(RESULT_STAT, 'epoch_%d' % eval_epoch)


    ## Main script for test phase 
    MSE_  = torch.zeros((args.future_frames), dtype = torch.float32).cuda()
    PSNR_ = torch.zeros((args.future_frames), dtype = torch.float32).cuda()
    SSIM_ = torch.zeros((args.future_frames), dtype = torch.float32).cuda()
    PIPS_ = torch.zeros((args.future_frames), dtype = torch.float32).cuda()

    with torch.no_grad():
        model.eval()
        
        samples = 0
        for it, frames in enumerate(test_loader):
            samples += total_batch_size

            if args.img_channels == 1:
                frames = torch.mean(frames, dim = -1, keepdim = True)

            if img_resize:
                frames_ = frames.cpu().numpy()
                frames = np.zeros((batch_size, total_frames, 
                    args.img_height_u, args.img_width_u, args.img_channels), dtype = np.float32)

                for b in range(batch_size):
                    for t in range(total_frames):
                        frames[b, t] = skimage.transform.resize(
                            frames_[b, t], (args.img_height_u, args.img_width_u))

                frames = torch.from_numpy(frames)

            # 5-th order: batch_size x total_frames x channels x height x width 
            frames = frames.permute(0, 1, 4, 2, 3).cuda()

            inputs = frames[:,  :args.input_frames]
            origin = frames[:, -args.future_frames:]

            pred = model(inputs, 
                input_frames  =  args.input_frames, 
                future_frames = args.future_frames, 
                output_frames = args.future_frames, 
                teacher_forcing = False, 
                checkpointing   = False)

            # clamp the output to [0, 1]
            pred = torch.clamp(pred, min = 0, max = 1)

            # save the first sample for each batch to the folder
            if (it + 1) % args.log_iterations == 0:
                # print(it, args.local_rank, 'log_iterations')
                b = args.local_rank
                input_0  = inputs[0, list_input_frames] 
                origin_0 = origin[0, list_future_frames]
                pred_0   =   pred[0, list_future_frames]

                # pad the input with zeros (if needed)
                if plot_input_frames < plot_future_frames:
                    input_0 = torch.cat([torch.zeros(
                        plot_future_frames - plot_input_frames, 
                        args.img_channels, args.img_height_u, args.img_width_u, 
                        device = "cuda"), input_0], dim = 0)

                img = torchvision.utils.make_grid(
                    torch.cat([input_0, origin_0, pred_0], 
                        dim = 0), nrow = plot_future_frames)
                
                RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
                RESULT_IMG = os.path.join(RESULT_DIR, "test_images_" + str(eval_epoch) + "_" + str(args.future_frames))
                RESULT_FILE = os.path.join(RESULT_IMG, "cmp_%d_%d.png" % (eval_epoch, samples + b))
                torchvision.utils.save_image(img, RESULT_FILE)

                if args.save_frames:
                    for t in range(0, args.input_frames):
                        input_ = inputs[0, t]
                        RESULT_FILE = os.path.join(RESULT_IMG, 
                            "inp_%d_%d_%d.png" % (eval_epoch, samples + b, t))
                        torchvision.utils.save_image(input_, RESULT_FILE)
                    print(RESULT_FILE)
                    
                    for t in range(0, args.future_frames):
                        origin_, pred_ = origin[0, t], pred[0, t]
                        RESULT_FILE = os.path.join(RESULT_IMG, 
                            "pred_%d_%d_%d.png" % (eval_epoch, samples + b, t))
                        torchvision.utils.save_image(pred_, RESULT_FILE)
                        RESULT_FILE = os.path.join(RESULT_IMG, 
                            "gt_%d_%d_%d.png"   % (eval_epoch, samples + b, t))
                        torchvision.utils.save_image(origin_, RESULT_FILE)

            # accumlate the statistics per frame
            for t in range(-args.future_frames, 0):
                origin_, pred_ = origin[:, t], pred[:, t]

                if args.img_channels == 1:
                    origin_ = origin_.repeat([1, 3, 1, 1])
                    pred_   =   pred_.repeat([1, 3, 1, 1])

                dist = PSmodel(origin_, pred_)
                PIPS_[t] += torch.sum(dist).item()

            origin = origin.permute(0, 1, 3, 4, 2).cpu().numpy()
            pred   =   pred.permute(0, 1, 3, 4, 2).cpu().numpy()

            for t in range(-args.future_frames, 0):
                for i in range(batch_size):
                    origin_, pred_ = origin[i, t], pred[i, t]

                    if args.img_channels == 1:
                        origin_ = np.squeeze(origin_, axis = -1)
                        pred_   = np.squeeze(pred_,   axis = -1)

                    MSE_[t]  += skimage.metrics.mean_squared_error(origin_, pred_)
                    PSNR_[t] += skimage.metrics.peak_signal_noise_ratio(origin_, pred_)
                    SSIM_[t] += skimage.metrics.structural_similarity(origin_, pred_, 
                        multichannel = args.img_channels > 1)

            if args.distributed:
                MSE  = reduce_tensor(MSE_,  reduce_sum = True) / samples
                PSNR = reduce_tensor(PSNR_, reduce_sum = True) / samples
                SSIM = reduce_tensor(SSIM_, reduce_sum = True) / samples
                PIPS = reduce_tensor(PIPS_, reduce_sum = True) / samples
            else:
                MSE  = MSE_  / samples
                PSNR = PSNR_ / samples
                SSIM = SSIM_ / samples
                PIPS = PIPS_ / samples

            if ((it + 1) % 50 == 0 or it + 1 == len(test_loader)) and args.local_rank == 0:
                print((it + 1) * total_batch_size, '/', test_samples,
                    ": MSE:  ", torch.mean(MSE ).cpu().item() * 1e3,
                    "; PSNR: ", torch.mean(PSNR).cpu().item(), 
                    "; SSIM: ", torch.mean(SSIM).cpu().item(), 
                    ";LPIPS: ", torch.mean(PIPS).cpu().item())

        if args.distributed:
            MSE  = reduce_tensor(MSE_,  reduce_sum = True) / test_samples
            PSNR = reduce_tensor(PSNR_, reduce_sum = True) / test_samples
            SSIM = reduce_tensor(SSIM_, reduce_sum = True) / test_samples
            PIPS = reduce_tensor(PIPS_, reduce_sum = True) / test_samples
        else:
            MSE  = MSE_  / test_samples
            PSNR = PSNR_ / test_samples
            SSIM = SSIM_ / test_samples
            PIPS = PIPS_ / test_samples

        MSE_AVG  = torch.mean(MSE ).cpu().item()
        PSNR_AVG = torch.mean(PSNR).cpu().item()
        SSIM_AVG = torch.mean(SSIM).cpu().item()
        PIPS_AVG = torch.mean(PIPS).cpu().item()

        if args.local_rank == 0:
            print("Epoch \t{} \tMSE: \t{} (x1e-3) \tPSNR: \t{} \tSSIM: \t{} \tLPIPS: \t{}".format(
                eval_epoch, 1e3 * MSE_AVG, PSNR_AVG, SSIM_AVG, PIPS_AVG))

            np.savez(RESULT_STAT, 
                MSE   =   MSE.cpu().numpy(), PSNR  =  PSNR.cpu().numpy(), 
                SSIM  =  SSIM.cpu().numpy(), PIPS  =  PIPS.cpu().numpy())
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Conv-TT-LSTM Test")

    ## Devices (Single GPU / Distributed computing)

    # whether to use distributed computing
    parser.add_argument('--use-distributed', dest = "distributed", 
        action = 'store_true',  help = 'Use distributed computing in training.')
    parser.add_argument( '--no-distributed', dest = "distributed", 
        action = 'store_false', help = 'Use single process (GPU) in training.')
    parser.set_defaults(distributed = True)

    parser.add_argument('--use-apex', dest = 'use_apex', 
        action = 'store_true',  help = 'Use DistributedDataParallel from apex.')
    parser.add_argument( '--no-apex', dest = 'use_apex', 
        action = 'store_false', help = 'Use DistributedDataParallel from nn.distributed.')
    parser.set_defaults(use_apex = False)

    parser.add_argument('--worker', default = 4, type = int, 
        help = 'number of workers for DataLoader.')

    # arguments for distributed computing 
    parser.add_argument('--local_rank', default = 0, type = int)
    parser.add_argument( '--node_rank', default = 0, type = int)


    ## Data format (batch_size x time_steps x height x width x channels)

    # batch size (0) 
    parser.add_argument('--batch-size', default = 16, type = int,
        help = 'The total batch size in each test iteration.')
    parser.add_argument('--log-iterations', default = 10, type = int,
        help = 'Log the test video every log_iterations.')

    # frame split (1)
    parser.add_argument('--input-frames',  default = 10, type = int,
        help = 'The number of input frames to the model.')
    parser.add_argument('--future-frames', default = 10, type = int,
        help = 'The number of predicted frames of the model.')
    
    parser.add_argument('--save-frames', dest = 'save_frames', 
        action = 'store_true',  help = 'Save frames for GIF')
    parser.add_argument('--log-frames', default = 1, type = int, 
        help = 'Log the frames every log_frames.')

    # frame format (2, 3, 4)
    parser.add_argument('--img-channels', default =  1, type = int, 
        help = 'The number of channels in each video frame.')

    parser.add_argument('--img-height',   default = 120, type = int, 
        help = 'The image height of each video frame.')
    parser.add_argument('--img-width',    default = 120, type = int, 
        help = 'The image width  of each video frame.')

    parser.add_argument('--img-height-u', default = 128, type = int, 
        help = 'The image height of each upsampled frame.')
    parser.add_argument('--img-width-u',  default = 128, type = int, 
        help = 'The image width  of each upsampled frame.')

    ## Models (Conv-LSTM or Conv-TT-LSTM)

    # model name (with time stamp as suffix)
    parser.add_argument('--model-name',  default = "test_0000", type = str,
        help = 'The model name is used to create the folder names.')

    # model type and size (depth and width) 
    parser.add_argument('--model', default = 'convlstm', type = str,
        help = 'The model is \"convlstm\", \"convttlstm\", \"ttconvlstm\", or \"orthoconvrnn\".')
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
    parser.add_argument('--model-ranks',  default = 8, type = int, 
        help = 'The tensor rank of the convolutional tensor-train LSTMs.')

    # parameters of the orthogonal convolutional recurrent layers
    parser.add_argument('--use-orthogonal', dest = "orthogonal", action = 'store_true',
        help = "Use orthogonal convoltion in the recurrent layer.")
    parser.add_argument( '--no-orthogonal', dest = "orthogonal", action = 'store_false',
        help = "Use standard convolution in the recurrent layer.")
    parser.set_defaults(use_bias = True)

    parser.add_argument('--ortho-init', default = "torus", type = str, 
        help = "The initialization method for the orthogonal convolutional kernels.")

    parser.add_argument('--activation', default = "modrelu", type = str, 
        "The activation function for the recurrent layer.")
    
    # parameters of the convolutional operations
    parser.add_argument('--kernel-size', default = 5, type = int,
        help = "The kernel size of the convolutional operations.")

    parser.add_argument('--use-bias', dest = "use_bias", 
        action = 'store_true',  help = "Add bias term after each convoltional operation.")
    parser.add_argument( '--no-bias', dest = "use_bias", 
        action = 'store_false', help = "Do not add bias after the convolutional operations.")
    parser.set_defaults(use_bias = True)

    ## Dataset (Input)
    parser.add_argument('--dataset', default = "MNIST", type = str, 
        help = 'The dataset name. (Options: KTH, MNIST)')
    parser.add_argument('--data-base', default = '../data/', type = str, 
        help = "The base path to the datasets.")
    parser.add_argument('--data-path', default = 'default', type = str,
        help = 'The path to the dataset folder.')

    parser.add_argument('--test-data-file', default = 'test', type = str, 
        help = 'Name of the folder/file for test set.')
    parser.add_argument('--test-samples', default = 5000, type = int, 
        help = 'Number of samples in test dataset.')

    ## Results and Models (Output)
    parser.add_argument('--output-base', default = './checkpoints/', type = str, 
        help = "The base path to the outputs.")
    parser.add_argument('--output-path', default = 'default', type = str,
        help = "The path to the folder storing the outputs (models and results).")

    ## Evaluation
    parser.add_argument('--eval-auto', dest = 'eval_auto', 
        action = 'store_true',  help = 'Evaluate the best or the last model.')
    parser.add_argument('--eval-spec', dest = 'eval_auto', 
        action = 'store_false', help = 'Evaluate the model of specified epoch')
    parser.set_defaults(eval_auto = True)

    # if eval_auto is True (--eval-auto)
    parser.add_argument('--eval-best', dest = 'eval_best', 
        action = 'store_true',  help = 'Evaluate the best model (in term of validation loss).')
    parser.add_argument('--eval-last', dest = 'eval_best', 
        action = 'store_false', help = 'Evaluate the last model (in term of training epoch).')
    parser.set_defaults(eval_best = True)

    # if eval_best is True (--eval-best)
    parser.add_argument('--eval-best-ssim', dest = 'eval_best_ssim', 
        action = 'store_true',  help = 'Evaluate the best model (in term of validation SSIM).')
    parser.add_argument('--eval-best-loss', dest = 'eval_best_ssim', 
        action = 'store_false', help = 'Evaluate the best model (in term of validation LOSS).')
    parser.set_defaults(eval_best_ssim = False)

    # if eval_auto is False (--eval-spec)
    parser.add_argument('--eval-epoch', default = 100, type = int, 
        help = 'Evaluate the model of specified epoch.')

    main(parser.parse_args())
