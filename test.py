import argparse
import os
import shutil
import sys

import torch
import torch.nn as nn
from autoattack import AutoAttack
from torchvision.models import resnet18, resnet50

from data import get_dataset
from models import Diffusion_Coefficients, Posterior_Coefficients
from train.train_utils import G_D_models, set_seeds


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


def load_classifiers(args, device):
    if args.dataset == "cifar10":
        classifier1 = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True
        ).to(device)
        classifier2 = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
        ).to(device)
    elif args.dataset == "GTSRB":
        netC18 = resnet18(pretrained=False)  # not using pretrained weights
        num_ftrs = netC18.fc.in_features
        netC18.fc = nn.Linear(num_ftrs, 43)
        netC18 = netC18.to(device)
        saved_model_classifier_18 = args.classifier_dir + "/netC_resnet18.pth"
        stateC = torch.load(saved_model_classifier_18)
        netC18.load_state_dict(stateC)
        classifier2 = netC18

        netC50 = resnet50(pretrained=False)  # not using pretrained weights
        num_ftrs = netC50.fc.in_features
        netC50.fc = nn.Linear(num_ftrs, 43)
        netC50 = netC50.to(device)
        saved_model_classifier_50 = args.classifier_dir + "/netC_resnet50.pth"
        stateC = torch.load(saved_model_classifier_50)
        netC50.load_state_dict(stateC)
        classifier1 = netC50

    return classifier1, classifier2


class Robust(nn.Module):
    def __init__(self, classifier, purifier, normalized=True):
        super().__init__()
        self.classifier = classifier
        self.purifier = purifier
        self.normalized = normalized

    def forward(self, real):

        x = self.purifier(real)

        if self.normalized:
            x_c = 2 * x - 1
        else:
            x_c = x

        out = self.classifier(x_c)
        return out


def test(args):
    set_seeds(args.seed)

    dataset = get_dataset(args.dataset, args.data_dir, args.image_size, mode="test")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(device)
    args.device = device
    print(args)

    saved_model = args.saved_generation

    netG, _ = G_D_models(args, device)
    exp = args.exp
    state = torch.load(saved_model)

    netG.load_state_dict(state)

    classifier1, classifier2 = load_classifiers(args, device)

    coeff = Diffusion_Coefficients(
        device, BETA_MAX=args.beta_max, BETA_MIN=args.beta_min
    )
    pos_coeff = Posterior_Coefficients(
        device, BETA_MAX=args.beta_max, BETA_MIN=args.beta_min
    )

    purifier1 = Purifier(netG, coeff, pos_coeff, args)

    normalize = False if args.dataset == "GTSRB" else True
    robust = Robust(classifier1, purifier1, normalized=normalize)

    if args.rand_attack:
        adversary_resnet_robust = AutoAttack(
            robust, norm="Linf", eps=8 / 255, version="rand", device=device
        )
        adversary_resnet_classifier1 = AutoAttack(
            classifier1, norm="Linf", eps=8 / 255, version="rand", device=device
        )
        adversary_resnet_classifier2 = AutoAttack(
            classifier2, norm="Linf", eps=8 / 255, version="rand", device=device
        )
    else:

        adversary_resnet_robust = AutoAttack(
            robust,
            norm="Linf",
            eps=8 / 255,
            version="custom",
            attacks_to_run=["apgd-ce"],
            device=device,
        )
        adversary_resnet_robust.apgd.n_restarts = 1
        adversary_resnet_robust.fab.n_restarts = 1
        adversary_resnet_robust.apgd_targeted.n_restarts = 1
        adversary_resnet_robust.fab.n_target_classes = 9
        adversary_resnet_robust.apgd_targeted.n_target_classes = 9
        adversary_resnet_robust.square.n_queries = 5000

        adversary_resnet_classifier1 = AutoAttack(
            classifier1,
            norm="Linf",
            eps=8 / 255,
            version="custom",
            attacks_to_run=["apgd-ce"],
            device=device,
        )
        adversary_resnet_classifier1.apgd.n_restarts = 1
        adversary_resnet_classifier1.fab.n_restarts = 1
        adversary_resnet_classifier1.apgd_targeted.n_restarts = 1
        adversary_resnet_classifier1.fab.n_target_classes = 9
        adversary_resnet_classifier1.apgd_targeted.n_target_classes = 9
        adversary_resnet_classifier1.square.n_queries = 5000

        adversary_resnet_classifier2 = AutoAttack(
            classifier2,
            norm="Linf",
            eps=8 / 255,
            version="custom",
            attacks_to_run=["apgd-ce"],
            device=device,
        )
        adversary_resnet_classifier2.apgd.n_restarts = 1
        adversary_resnet_classifier2.fab.n_restarts = 1
        adversary_resnet_classifier2.apgd_targeted.n_restarts = 1
        adversary_resnet_classifier2.fab.n_target_classes = 9
        adversary_resnet_classifier2.apgd_targeted.n_target_classes = 9
        adversary_resnet_classifier2.square.n_queries = 5000

    robust.eval()
    correct_clean = 0
    correct_white = 0
    correct_gray = 0
    correct_black = 0

    total = 0
    c = 0

    with open(args.name, "w") as file:
        original_stdout = sys.stdout
        sys.stdout = file

        print(args)
        print(saved_model)
        print(device)
        for inputs, labels in data_loader:
            c += 1
            inputs, labels = inputs.to(device), labels.to(
                device
            )  # Move data to the specified device

            x_adv_white = adversary_resnet_robust.run_standard_evaluation(
                inputs, labels
            )
            gray_inputs = 0.5 * inputs + 0.5 if normalize else inputs
            x_adv_gray = adversary_resnet_classifier1.run_standard_evaluation(
                gray_inputs, labels
            )
            x_adv_black = adversary_resnet_classifier2.run_standard_evaluation(
                gray_inputs, labels
            )

            outputs_clean = robust(inputs)
            _, predicted_clean = torch.max(outputs_clean, 1)

            outputs_white = robust(x_adv_white)
            _, predicted_white = torch.max(outputs_white, 1)

            x_adv_gray = 2 * x_adv_gray - 1 if normalize else x_adv_gray
            outputs_gray = robust(x_adv_gray)
            _, predicted_gray = torch.max(outputs_gray, 1)

            x_adv_black = 2 * x_adv_black - 1 if normalize else x_adv_black
            outputs_black = robust(x_adv_black)
            _, predicted_black = torch.max(outputs_black, 1)

            total += labels.size(0)

            correct_clean += (predicted_clean == labels).sum().item()
            correct_white += (
                torch.logical_and(predicted_white == labels, predicted_clean == labels)
                .sum()
                .item()
            )
            correct_gray += (predicted_gray == labels).sum().item()
            correct_black += (
                torch.logical_and(predicted_black == labels, predicted_clean == labels)
                .sum()
                .item()
            )

            print(correct_clean / total)
            print(correct_white / total)
            print(correct_gray / total)
            print(correct_black / total)

            print(c)
            if total > 5000:
                break

        accuracy_clean = 100 * correct_clean / total
        accuracy_white = 100 * correct_white / total
        accuracy_gray = 100 * correct_gray / total
        accuracy_black = 100 * correct_black / total

        print(f"clean accuracy is : {accuracy_clean}")
        print(f"white box robust accuracy is : {accuracy_white}")
        print(f"gray box robust accuracy is : {accuracy_gray}")
        print(f"black box robust accuracy is : {accuracy_black}")


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

    parser.add_argument("--device", type=int, default=0)

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
    parser.add_argument("--batch_size", type=int, default=48, help="input batch size")
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
