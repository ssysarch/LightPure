import torch
import torch.nn as nn
import torch.optim as optim

import shutil
import os
from models import NCSNpp, Discriminator_small, Discriminator_large
from .EMA import EMA


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def G_D_models(args, device):
    G = NCSNpp(args).to(device)
    # TODO: Discriminator_large for other datasets
    D = Discriminator_small(
        nc=2 * args.num_channels,
        ngf=args.ngf,
        act=nn.LeakyReLU(0.2),
    ).to(device)
    return G, D


def G_D_optimizers(args, netG, netD):
    optimizerG = optim.Adam(
        netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2)
    )
    optimizerD = optim.Adam(
        netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2)
    )

    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    return optimizerG, optimizerD


def G_D_schedulers(num_epoch, optimizerG, optimizerD):
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerG, num_epoch, eta_min=1e-5
    )
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerD, num_epoch, eta_min=1e-5
    )

    return schedulerG, schedulerD


def load_checkpoint(
    ckpt, device, netG, netD, optimizerG, optimizerD, schedulerG, schedulerD
):
    checkpoint = torch.load(ckpt, map_location=device)
    init_epoch = checkpoint["epoch"]

    netG.load_state_dict(checkpoint["netG_dict"])
    optimizerG.load_state_dict(checkpoint["optimizerG"])
    schedulerG.load_state_dict(checkpoint["schedulerG"])
    # load D
    netD.load_state_dict(checkpoint["netD_dict"])
    optimizerD.load_state_dict(checkpoint["optimizerD"])
    schedulerD.load_state_dict(checkpoint["schedulerD"])
    global_step = checkpoint["global_step"]
    print("=> loaded checkpoint (epoch {})".format(checkpoint["epoch"]))

    return global_step, init_epoch



