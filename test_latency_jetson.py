import argparse
import torch
from piq import ssim
from autoattack import AutoAttack
from contextlib import redirect_stdout
import sys
import os
from robustbench.data import load_cifar10
import time
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from robustbench.data import load_cifar10
from torchvision.models import resnet50
from torchvision import transforms
from torchvision.models import resnet18

from data import GTSRB
from train.train_utils import (
    set_seeds,
    G_D_models,
    G_D_optimizers,
    G_D_schedulers,
    load_checkpoint,
)

from models import  Diffusion_Coefficients, Posterior_Coefficients
import shutil
from tensorboardX import SummaryWriter

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


class Purifier(nn.Module):
    def __init__(self, generator, coeff, pos_coeff, args):
        super().__init__()
        self.generator = generator
        self.coeff = coeff
        self.pos_coeff = pos_coeff
        self.args = args

    def forward(self, real):
        x_1, x_2 = self.coeff.q_sample_pairs(real)
        x = x_2

        latent_z = torch.randn(real.size(0), self.args.nz, device=args.device)
        x_0_predict = self.generator(x, latent_z)

        return x_0_predict


class Robust(nn.Module):
    def __init__(self, classifier, purifier, normilized = True):
        super().__init__()
        self.classifier = classifier
        self.purifier = purifier
        self.normilized = normilized


    def forward(self, real):



        x = self.purifier(real)

        if self.normilized:
            x_c  = 2*x -1
        else:
            x_c = x

        out = self.classifier(x_c)
        return out

def test(args):
    set_seeds(args.seed)

    dataset = GTSRB(
        root_dir='/home/vinecnt/Desktop/archive',
        train=True,
        transform=transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
            ]
        ),
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )



    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
    print(device)
    args.device = device
    print(args)

    saved_model = args.saved_generation



    netG, _ = G_D_models(args, device)
    exp = args.exp
    state = torch.load(saved_model)

    netG.load_state_dict(state)



    netC18 = resnet18(pretrained=False)  # not using pretrained weights
    num_ftrs = netC18.fc.in_features
    netC18.fc = nn.Linear(num_ftrs, 43)
    netC18 = netC18.to(device)
    saved_model_classifier_18 = '/home/vinecnt/Downloads/netC_64.pth'
    stateC = torch.load(saved_model_classifier_18)
    netC18.load_state_dict(stateC)
    classifier1 = netC18

    netC50 = resnet50(pretrained=False)  # not using pretrained weights
    num_ftrs = netC50.fc.in_features
    netC50.fc = nn.Linear(num_ftrs, 43)
    netC50 = netC50.to(device)
    saved_model_classifier_50 = '/home/vinecnt/Downloads/netC_60.pth'
    stateC = torch.load(saved_model_classifier_50)
    netC50.load_state_dict(stateC)
    classifier2 = netC50


    coeff = Diffusion_Coefficients(device, BETA_MAX=args.beta_max , BETA_MIN=args.beta_min)
    pos_coeff = Posterior_Coefficients(device, BETA_MAX=args.beta_max , BETA_MIN=args.beta_min)

    purifier1 = Purifier(netG, coeff, pos_coeff, args)
    robust_n = Robust(classifier1, purifier1, normilized=True)
    robust_un = Robust(classifier1, purifier1, normilized=False)





    robust_un.eval()
    robust_n.eval()

    latency_purifier = 0
    latency_classifier1= 0
    latency_classifier2=0



    total = 0
    c = 0


    classifier1_d = classifier1.to(device)
    classifier2_d = classifier2.to(device)

    print(args)
    print(saved_model)
    print(device)

    classifier1.eval()
    classifier1.eval()


    classifier1_d.eval()
    classifier2_d.eval()

    torch.no_grad()
    
    x_val, y_val = next(iter(data_loader))

    for idx in range(0, len(x_val), 1):
        c += 1
        inputs, labels = x_val[idx:idx+1].to(device), y_val[idx:idx+1].to(device)  # Move data to the specified device
        

        start = time.time()
        outputs_clean = robust_un(inputs )
        stop = time.time()
        print(stop - start)
        latency_purifier += stop - start





        start = time.time()
        c = classifier1_d(inputs )
        stop = time.time()
        print(stop - start)
        latency_classifier1 += stop - start


        start = time.time()
        c = classifier2_d(inputs )
        stop = time.time()
        print(stop - start)
        latency_classifier2 += stop - start




        total += labels.size(0)



        print(c)
        if total > 1000:
            break
    print(f'latency purifier : {latency_purifier/total}')
    print(f'latency classifeir1 : {latency_classifier1/total}')
    print(f'latency classifier2 : {latency_classifier2/total}')
    print(len(x_val), total)

    # for inputs, labels in data_loader:
    #     c += 1
    #     inputs, labels = inputs.to(device), labels.to(device)  # Move data to the specified device


    #     start = time.time()
    #     outputs_clean = robust_un(inputs )
    #     stop = time.time()
    #     print(stop - start)
    #     latency_purifier += stop - start

    #     print("SIZE: ", inputs.shape)


    #     start = time.time()
    #     c = classifier1_d(inputs )
    #     stop = time.time()
    #     print(stop - start)
    #     latency_classifier1 += stop - start


    #     start = time.time()
    #     c = classifier2_d(inputs )
    #     stop = time.time()
    #     print(stop - start)
    #     latency_classifier2 += stop - start




    #     total += labels.size(0)



    #     print(c)
    #     if total > 1000:
    #         break




    print(f'latency purifier : {latency_purifier/total}')
    print(f'latency classifeir1 : {latency_classifier1/total}')
    print(f'latency classifier2 : {latency_classifier2/total}')





if __name__ == "__main__":
    print("starting")

    parser = argparse.ArgumentParser("ddgan parameters")
    parser.add_argument(
        "--seed", type=int, default=1024, help="seed used for initialization"
    )

    parser.add_argument("--ckpt", default=None, help="path to checkpoint")

    parser.add_argument("--name", default=None, help="name of text")
    parser.add_argument("--saved_generation", default=None, help="path of generation")

    parser.add_argument("--image_size", type=int, default=32, help="size of image")
    parser.add_argument("--num_channels", type=int, default=3, help="channel of image")
    parser.add_argument(
        "--centered", action="store_false", default=True, help="-1,1 scale"
    )
    parser.add_argument("--use_geometric", action="store_true", default=False)
    parser.add_argument(
        "--beta_min", type=float, default=0.1, help="beta_min for diffusion"
    )
    parser.add_argument(
        "--beta_max", type=float, default=20.0, help="beta_max for diffusion"
    )

    parser.add_argument(
        "--device", type=int, default=0
    )

    parser.add_argument(
        "--num_channels_dae",
        type=int,
        default=128,
        help="number of initial channels in denosing model",
    )
    parser.add_argument(
        "--n_mlp", type=int, default=3, help="number of mlp layers for z"
    )
    parser.add_argument(
        "--ch_mult",
        nargs="+",
        default=[1, 2, 2, 2],
        type=int,
        help="channel multiplier",
    )
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="number of resnet blocks per scale",
    )
    parser.add_argument(
        "--attn_resolutions", default=(16,), help="resolution of applying attention"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument(
        "--resamp_with_conv",
        action="store_false",
        default=True,
        help="always up/down sampling with conv",
    )
    parser.add_argument(
        "--conditional", action="store_false", default=True, help="noise conditional"
    )
    parser.add_argument("--fir", action="store_false", default=True, help="FIR")
    parser.add_argument("--fir_kernel", default=[1, 3, 3, 1], help="FIR kernel")
    parser.add_argument(
        "--skip_rescale", action="store_false", default=True, help="skip rescale"
    )
    parser.add_argument(
        "--resblock_type",
        default="biggan",
        help="tyle of resnet block, choice in biggan and ddpm",
    )
    parser.add_argument(
        "--progressive",
        type=str,
        default="none",
        choices=["none", "output_skip", "residual"],
        help="progressive type for output",
    )
    parser.add_argument(
        "--progressive_input",
        type=str,
        default="residual",
        choices=["none", "input_skip", "residual"],
        help="progressive type for input",
    )
    parser.add_argument(
        "--progressive_combine",
        type=str,
        default="sum",
        choices=["sum", "cat"],
        help="progressive combine method.",
    )

    parser.add_argument(
        "--fourier_scale", type=float, default=16.0, help="scale of fourier transform"
    )
    parser.add_argument("--not_use_tanh", action="store_true", default=False)
    parser.add_argument("--rand_attack", action="store_true", default=False)

    # geenrator and training
    parser.add_argument(
        "--exp", default="experiment_cifar_default", help="name of experiment"
    )
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--nz", type=int, default=100)

    parser.add_argument("--z_emb_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32, help="input batch size")
    parser.add_argument("--num_epoch", type=int, default=1200)
    parser.add_argument("--ngf", type=int, default=64)

    parser.add_argument("--lr_g", type=float, default=1.5e-4, help="learning rate g")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="learning rate d")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.9, help="beta2 for adam")
    parser.add_argument("--no_lr_decay", action="store_true", default=False)

    parser.add_argument(
        "--use_ema", action="store_true", default=False, help="use EMA or not"
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.9999, help="decay rate for EMA"
    )

    parser.add_argument("--r1_gamma", type=float, default=0.05, help="coef for r1 reg")
    parser.add_argument(
        "--lazy_reg", type=int, default=None, help="lazy regulariation."
    )

    parser.add_argument("--save_content", action="store_true", default=True)
    parser.add_argument(
        "--save_content_every",
        type=int,
        default=10,
        help="save content for resuming every x epochs",
    )
    parser.add_argument(
        "--save_ckpt_every", type=int, default=25, help="save ckpt every x epochs"
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="data directory")
    parser.add_argument(
        "--ckpt_dir", type=str, default="./saved_model", help="checkpoint directory"
    )
    parser.add_argument("--config", type=str, default=None, help="config file")

    args = parser.parse_args()
    if args.config is not None:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            for k, v in config.items():
                setattr(args, k, v)

    os.environ["MASTER_PORT"] = "29501"
    test(args)
