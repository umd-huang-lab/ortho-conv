import argparse
import os
import json
import shutil
import random
from itertools import islice

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss

from utils.datasets import check_dataset
from utils.models import Glow


def check_manual_seed(seed):
    """
    Specify the random seed for training.

    Argument:
    ---------
    seed: int or None 
        The random seed for training.
        Note: The seed is randomly set to [1, 10000] if None is supplied.

    """
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using seed: {seed}".format(seed=seed))


def compute_loss(nll, reduction = "mean"):
    """
    Compute the NLL loss for training.
    
    Arguments:
    ----------
    nll: 

    reduction: 

    Return:
    -------
    losses:
    

    """
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}

    elif reduction == "sum":
        losses = {"nll": torch.sum(nll)}

    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


def compute_loss_y(nll, y_logits, y_weight, y, multi_class, reduction = "mean"):
    """
    """
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}

    elif reduction == "sum":
        losses = {"nll": torch.sum(nll)} 

    elif reduction == "none":
        losses = {"nll": nll}

    if multi_class:
        y_logits = torch.sigmoid(y_logits)
        loss_classes = F.binary_cross_entropy_with_logits(
            y_logits, y, reduction = reduction)

    else: # if not multi_class:
        loss_classes = F.cross_entropy(
            y_logits, torch.argmax(y, dim = 1), reduction=reduction)

    losses["loss_classes"] = loss_classes
    losses["total_loss"] = losses["nll"] + y_weight * loss_classes

    return losses


def main(
    dataset,
    dataroot,
    download,
    augment,
    batch_size,
    eval_batch_size,
    epochs,
    saved_checkpoint,
    seed,
    hidden_channels,
    num_scales,
    num_blocks,
    actnorm_scale,
    flow_permutation,
    ortho_ker_size,
    ortho_ker_init,
    LU_decomposed,
    flow_coupling,
    learn_top,
    y_condition,
    y_weight,
    max_grad_clip,
    max_grad_norm,
    lr,
    num_workers,
    cuda,
    n_init_batches,
    output_dir,
    warm_epochs,
    save_epochs,
    eval_epochs
):

    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda"

    check_manual_seed(seed)

    image_shape, num_classes, train_dataset, test_dataset = \
        check_dataset(dataset, dataroot, augment, download)

    # Note: unsupported for now
    multi_class = False

    train_loader = data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        drop_last = True
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size = eval_batch_size,
        shuffle = False,
        num_workers = num_workers,
        drop_last = False
    )

    model = Glow(
        image_shape,
        hidden_channels,
        num_scales,
        num_blocks,
        actnorm_scale,
        flow_permutation,
        ortho_ker_size,
        ortho_ker_init,
        LU_decomposed,
        flow_coupling,
        num_classes,
        learn_top,
        y_condition,
    )

    model = model.to(device)
    optimizer = optim.Adamax(model.parameters(), lr = lr, weight_decay = 5e-5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
        lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warm_epochs))

    # Training routine
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        x = x.to(device)

        if y_condition:
            y = y.to(device)
            z, nll, y_logits = model(x, y)
            losses = compute_loss_y(nll, y_logits, y_weight, y, multi_class)
        else:
            z, nll, y_logits = model(x, None)
            losses = compute_loss(nll)

        losses["total_loss"].backward()

        if max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_clip)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_( model.parameters(), max_grad_norm)

        optimizer.step()

        return losses

    trainer = Engine(train_step)

    # evaluation routine 
    def eval_step(engine, batch):
        model.eval()

        x, y = batch
        x = x.to(device)

        with torch.no_grad():
            if y_condition:
                y = y.to(device)
                z, nll, y_logits = model(x, y)
                losses = compute_loss_y(
                    nll, y_logits, y_weight, y, multi_class, reduction = "none"
                )
            else:
                z, nll, y_logits = model(x, None)
                losses = compute_loss(nll, reduction = "none")

        return losses

    evaluator = Engine(eval_step)
    
    # checkpoint routine
    def global_step_transform(engine, EPOCH_COMPLETED):
        return engine.state.epoch

    checkpoint_handler = ModelCheckpoint(output_dir, "glow", 
        global_step_transform = global_step_transform, n_saved = 2, require_empty = False)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every = save_epochs),
        checkpoint_handler, {"model": model, "optimizer": optimizer})

    # specify monitoring metrics
    monitoring_metrics = ["total_loss"]
    RunningAverage(output_transform = lambda x: x["total_loss"]).attach(
        trainer, "total_loss"
    )

    # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
    Loss(
        lambda x, y: torch.mean(x),
        output_transform = lambda x: (
            x["total_loss"],
            torch.empty(x["total_loss"].shape[0]),
        ),
    ).attach(evaluator, "total_loss")

    if y_condition:
        monitoring_metrics.extend(["nll"])
        RunningAverage(output_transform = lambda x: x["nll"]).attach(trainer, "nll")

        # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
        Loss(
            lambda x, y: torch.mean(x),
            output_transform = lambda x: (x["nll"], torch.empty(x["nll"].shape[0])),
        ).attach(evaluator, "nll")

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names = monitoring_metrics)

    # load pre-trained model and optimizer (if provided)
    if saved_checkpoint:
        checkpoint = torch.load(saved_checkpoint)

        model.load_state_dict(checkpoint["model"])
        model.set_actnorm_init()

        optimizer.load_state_dict(checkpoint["optimizer"])

        file_name, ext = os.path.splitext(saved_checkpoint)
        resume_epoch = int(file_name.split("_")[-1])

        @trainer.on(Events.STARTED)
        def resume_training(engine):
            engine.state.epoch = resume_epoch
            engine.state.iteration = resume_epoch * len(engine.state.dataloader)

    @trainer.on(Events.STARTED)
    def init(engine):
        model.train()

        init_batches = []
        init_targets = []

        with torch.no_grad():
            for batch, target in islice(train_loader, None, n_init_batches):
                init_batches.append(batch)
                init_targets.append(target)

            init_batches = torch.cat(init_batches).to(device)

            assert init_batches.shape[0] == n_init_batches * batch_size

            if y_condition:
                init_targets = torch.cat(init_targets).to(device)
            else:
                init_targets = None

            model(init_batches, init_targets)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(engine):
        # adjust the learning rate
        scheduler.step()

        # evaluate the test set
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics

        losses = ", ".join([f"{key}: {value:.2f}" for key, value in metrics.items()])
        print(f"Validation results - Epoch: {engine.state.epoch} {losses}")

        # evaluate the training set
        if engine.state.epoch % eval_epochs == 0:
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics

            losses = ", ".join([f"{key}: {value:.2f}" for key, value in metrics.items()])
            print(f"Training results - Epoch: {engine.state.epoch} {losses}")

    timer = Timer(average = True)
    timer.attach(
        trainer,
        start  = Events.EPOCH_STARTED,
        resume = Events.ITERATION_STARTED,
        pause  = Events.ITERATION_COMPLETED,
        step   = Events.ITERATION_COMPLETED,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message(
            f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]"
        )
        timer.reset()

    trainer.run(train_loader, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type = str, default = "CIFAR10",
        choices = ["CIFAR10", "SVHN", "MNIST", "FashionMNIST"], 
        help = "The dataset used for training."
    )

    parser.add_argument("--dataroot", type = str, default = "./", 
        help = "The path to root folder of all datasets."
    )

    parser.add_argument("--download", action = "store_true", dest = "download",
        help = "Whether to download the specified dataset."
    )

    parser.add_argument("--no_augment", action = "store_false", dest = "augment", 
        help = "Whether to augment the images during training."
    )

    parser.add_argument("--num_scales", type = int, default = 3,
        help = "The number of scales in the network."
    )

    parser.add_argument("--num_blocks", type = int, default = 32, 
        help = "The number of blocks for each scale."
    )

    parser.add_argument("--hidden_channels", type = int, default = 512, 
        help = "The number of hidden channels in each block."
    )

    parser.add_argument("--actnorm_scale", type = float, default = 1.0, 
        help = "The scale for the activation normalization."
    )

    parser.add_argument("--flow_coupling", type = str, default = "affine",
        choices = ["additive", "affine"], 
        help = "The type of flow coupling."
    )

    parser.add_argument("--flow_permutation", type = str, default = "invconv",
        choices = ["invconv", "orthoconv", "shuffle", "reverse"], 
        help = "The type of flow permutation."
    )

    parser.add_argument("--ortho_ker_size", type = int, default = 3, 
        help = "The kernel size of orthogonal convolutional layers in flow permutation."
    )

    parser.add_argument("--ortho_ker_init", type = str, default = "uniform",
        choices = ["uniform", "identical", "reverse", "permutation"],
        help = "The initialization method of the orthogonal convolutional layers." 
    )

    parser.add_argument("--no_LU_decomposed", action = "store_false", dest = "LU_decomposed", 
        help = "Whether to LU-decompose the weights in the 1x1 invertible convolutional layers."
    )

    parser.add_argument("--no_learn_top", dest = "learn_top", 
        action = "store_false", help = "Do not train top layer (prior)"
    )

    parser.add_argument("--y_condition", action = "store_true", 
        help = "Whether to train the model conditioning on the class label."
    )

    parser.add_argument("--y_weight", type = float, default = 0.01, 
        help="Weight for class condition loss"
    )

    parser.add_argument("--max_grad_clip", type = float, default = 0,
        help = "Max gradient value (clip above - for off)",
    )

    parser.add_argument("--max_grad_norm", type = float, default = 0,
        help="Max norm of gradient (clip above - 0 for off)",
    )

    parser.add_argument("--num_workers", type = int, default = 6, 
        help = "number of data loading workers"
    )

    parser.add_argument("--batch_size", type = int, default = 64, 
        help = "The number of images in a batch during training."
    )

    parser.add_argument("--eval_batch_size", type = int, default = 512,
        help = "The number of images in a batch during evaluation."
    )

    parser.add_argument("--lr", type = float, default = 1e-4, 
        help = "The base learning rate for the AdaMax optimizer."
    )

    parser.add_argument("--epochs", type = int, default = 3000, 
        help = "The number of epochs to train the Glow model."
    )

    parser.add_argument("--save_epochs", type = int, default = 10, 
        help = "The number of epochs for each checkpoint saving."
    )

    parser.add_argument("--eval_epochs", type = int, default = 10, 
        help = "The number of epochs for each full evaluation (including training set).")

    parser.add_argument("--warm_epochs", type = float, default = 10,
        help = "The number of epochs for warmup training." # noqa
    )

    parser.add_argument("--n_init_batches", type = int, default = 8,
        help = "Number of batches to use for Act Norm initialisation",
    )

    parser.add_argument("--no_cuda", action = "store_false", dest = "cuda", 
        help = "Whether to use GPU or CPU for training."
    )

    parser.add_argument("--output_dir", default = "./outputs/",
        help = "Directory to output logs and model checkpoints",
    )

    parser.add_argument("--fresh", action = "store_true", 
        help = "Remove output directory before starting."
    )

    parser.add_argument("--saved_checkpoint", default = "",
        help = "The path to the checkpoint for continuing training.",
    )

    parser.add_argument("--seed", type = int, default = 0, 
        help = "Specify the random seed manually."
    )

    args = parser.parse_args()

    try:
        os.makedirs(args.output_dir)
    except FileExistsError:
        if args.saved_checkpoint:
            assert os.path.isdir(args.output_dir) 

        else:  
            if args.fresh:
                shutil.rmtree(args.output_dir)
                os.makedirs(args.output_dir)

            if (not os.path.isdir(args.output_dir)) or (
                len(os.listdir(args.output_dir)) > 0
            ):
                raise FileExistsError(
                    "Please provide a path to a non-existing or empty directory. Alternatively, pass the --fresh flag."  # noqa
                )

    kwargs = vars(args)
    del kwargs["fresh"]

    with open(os.path.join(args.output_dir, "hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys = True, indent = 4)

    main(**kwargs)
