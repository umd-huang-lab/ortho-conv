# evaluate a smoothed classifier on a dataset
import argparse
import datetime
import os
from time import time

from architectures import get_architecture
from core import Smooth
from datasets import get_dataset, DATASETS, get_num_classes
import torch


parser = argparse.ArgumentParser(description='Certify many examples')

# dataset
parser.add_argument('--dataset', type=str, choices=DATASETS)
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")

# load base classifier
# parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument('--noise_sd', default=0.0, type=float, help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--seed', type=int, default=2, metavar='S', help='random seed (default: 1)')

# randomized smoothing
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")

args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    training_speficiation = args.mode + '_std_' + str(args.noise_sd) + "_lr_" + str(args.lr) + '_seed_' + str(args.seed)
    load_dir = os.path.join(args.model_dir, args.dataset, args.mode, training_speficiation)
    checkpoint = torch.load(os.path.join(load_dir, 'checkpoint.pth.tar'))
    
    torch.manual_seed(args.seed)
    if args.dataset == 'mnist':
        if args.mode == 'full':
            base_classifier = LeNet_orth_full()
        elif args.mode == 'partial':
            base_classifier = LeNet_orth_partial()
        elif args.mode == 'normal':
            base_classifier = LeNet()

    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    save_file = os.path.joint(load_dir, 'out.txt')
    f = open(save_file, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
