import argparse
import torch
from piq import ssim
import torch.optim as optim
import os

import torch.nn.functional as F
import torchvision
from torchvision import transforms

from data import GTSRB
from train.train_utils import (
    set_seeds,
    G_D_models,
    G_D_optimizers,
    G_D_schedulers,
    load_checkpoint,
)
import torch.nn as nn

from models import Diffusion_Coefficients, Posterior_Coefficients
import shutil
from tensorboardX import SummaryWriter

from torchvision.models import resnet50
def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def train(args):
    set_seeds(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset = GTSRB(
        root_dir='/data/GTSRB/',
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
    netC = resnet50(pretrained=False)  # not using pretrained weights
    num_ftrs = netC.fc.in_features
    netC.fc = nn.Linear(num_ftrs, 43)
    netC = netC.to(device)


    exp = args.exp
    exp_path = os.path.join(args.ckpt_dir, "{}".format(args.dataset), exp)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(netC.parameters(), lr=0.001, momentum=0.9)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        copy_source(__file__, exp_path)

    logger = SummaryWriter(os.path.join(exp_path, "logs"))



    print(args)

    torch.save(args, f"{exp_path}/args.pth")

    for epoch in range(0, args.num_epoch + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (x, y) in enumerate(data_loader):

            optimizer.zero_grad()
            real_data = x.to(device, non_blocking=True)
            y = y.to(device)

            outputs = netC(real_data)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_acc:.2f}%')
        logger.add_scalar("loss", epoch_loss, epoch)
        logger.add_scalar("accuracy",epoch_acc, epoch)

        if epoch % args.save_ckpt_every == 0:
            torch.save(
                netC.state_dict(),
                os.path.join(exp_path, "netC_{}.pth".format(epoch)),
            )



if __name__ == "__main__":
    print("starting")

    parser = argparse.ArgumentParser("ddgan parameters")
    parser.add_argument(
        "--seed", type=int, default=1024, help="seed used for initialization"
    )

    parser.add_argument("--ckpt", default=None, help="path to checkpoint")

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
    parser.add_argument("--ssim", action="store_true", default=False)

    # geenrator and training
    parser.add_argument(
        "--exp", default="experiment_cifar_default", help="name of experiment"
    )
    parser.add_argument("--dataset", default="GTSRB", help="name of dataset")
    parser.add_argument("--nz", type=int, default=100)

    parser.add_argument("--z_emb_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256, help="input batch size")
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
        "--save_ckpt_every", type=int, default=5, help="save ckpt every x epochs"
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
    train(args)
