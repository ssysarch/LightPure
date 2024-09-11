import argparse
import torch
from piq import ssim

import os

import torch.nn.functional as F
import torchvision

from data import get_dataset
from train.train_utils import (
    set_seeds,
    G_D_models,
    G_D_optimizers,
    G_D_schedulers,
    load_checkpoint,
)

from models import Diffusion_Coefficients, Posterior_Coefficients
import shutil
from tensorboardX import SummaryWriter

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def train(args):
    set_seeds(args.seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset = get_dataset(args.dataset, args.data_dir, args.image_size, mode="train")


    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    netG, netD = G_D_models(args, device)
    exp = args.exp
    exp_path = os.path.join(args.ckpt_dir, "{}".format(args.dataset), exp)

    optimizerG, optimizerD = G_D_optimizers(args, netG, netD)
    schedulerG, schedulerD = G_D_schedulers(args.num_epoch, optimizerG, optimizerD)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        copy_source(__file__, exp_path)

    logger = SummaryWriter(os.path.join(exp_path, "logs"))


    if args.ckpt is not None and os.path.exists(args.ckpt):
        global_step, init_epoch = load_checkpoint(
            args.ckpt,
            device,
            netG,
            netD,
            optimizerG,
            optimizerD,
            schedulerG,
            schedulerD,
        )
    else:
        global_step, init_epoch = 0, 0

    print(args)
    coef = Diffusion_Coefficients(device, BETA_MAX=args.beta_max, BETA_MIN=args.beta_min)
    coef_pos = Posterior_Coefficients(device, BETA_MAX=args.beta_max, BETA_MIN=args.beta_min)
    torch.save(args, f"{exp_path}/args.pth")

    for epoch in range(init_epoch, args.num_epoch + 1):
        for iteration, (x, y) in enumerate(data_loader):
            for p in netD.parameters():
                p.requires_grad = True

            netD.zero_grad()

            real_data = x.to(device, non_blocking=True)

            x_1, x_2 = coef.q_sample_pairs(real_data)
            x_1.requires_grad = True

            # train with real
            D_real = netD(x_1, x_2.detach()).view(-1)

            errD_real = F.softplus(-D_real)
            errD_real = errD_real.mean()

            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:
                grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_1, create_graph=True
                )[0]
                grad_penalty = (
                    grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()

                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad_real = torch.autograd.grad(
                        outputs=D_real.sum(), inputs=x_1, create_graph=True
                    )[0]
                    grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()

                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

            latent_z = torch.randn(args.batch_size, args.nz, device=device)

            x_0_predict = netG(x_2.detach(), latent_z)
            x_pos_sample = coef_pos.sample_posterior(x_0_predict, x_2)

            output = netD(x_pos_sample, x_2.detach()).view(-1)

            errD_fake = F.softplus(output)
            errD_fake = errD_fake.mean()
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()

            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            x_1, x_2 = coef.q_sample_pairs(real_data)

            latent_z = torch.randn(args.batch_size, args.nz, device=device)

            x_0_predict = netG(x_2.detach(), latent_z)
            x_pos_sample = coef_pos.sample_posterior(x_0_predict, x_2)

            output = netD(x_pos_sample, x_2.detach()).view(-1)

            errG = F.softplus(-output)
            errG = errG.mean()

            if args.ssim:
                loss_g = 1 - ssim(torch.clamp(.5 * x_pos_sample+ .5, min=0, max=1) ,
                                torch.clamp(.5 * x_1 + .5, min=0, max=1), data_range=1.)
                errG += 5 * loss_g

            errG.backward()
            optimizerG.step()

            global_step += 1
            if iteration % 100 == 0:
                logger.add_scalar("G_loss", errG.item(), global_step)
                logger.add_scalar("D_loss", errD.item(), global_step)
                if args.ssim:
                    logger.add_scalar("loss_g", 5 * loss_g.item(), global_step)
                    logger.add_scalar("G_loss_pure",errG.item() -  5 * loss_g.item(), global_step)

                print(
                    "epoch {} iteration{}, G Loss: {}, D Loss: {}".format(
                        epoch, iteration, errG.item(), errD.item()
                    )
                )

        if not args.no_lr_decay:
            schedulerG.step()
            schedulerD.step()

        if epoch % 10 == 0:
            torchvision.utils.save_image(
                x_pos_sample,
                os.path.join(exp_path, "xpos_epoch_{}.png".format(epoch)),
                normalize=True,
            )
            torchvision.utils.save_image(
                x_0_predict,
                os.path.join(exp_path, "x_0_predict_epoch_{}.png".format(epoch)),
                normalize=True,
            )
            torchvision.utils.save_image(
                real_data,
                os.path.join(exp_path, "real_data_epoch_{}.png".format(epoch)),
                normalize=True,
            )
            torchvision.utils.save_image(
                x_1,
                os.path.join(exp_path, "x_1_epoch_{}.png".format(epoch)),
                normalize=True,
            )

            torchvision.utils.save_image(
                x_2,
                os.path.join(exp_path, "x_2_epoch_{}.png".format(epoch)),
                normalize=True,
            )



        if args.save_content:
            if epoch % args.save_content_every == 0:
                print("Saving content.")
                content = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "args": args,
                    "netG_dict": netG.state_dict(),
                    "optimizerG": optimizerG.state_dict(),
                    "schedulerG": schedulerG.state_dict(),
                    "netD_dict": netD.state_dict(),
                    "optimizerD": optimizerD.state_dict(),
                    "schedulerD": schedulerD.state_dict(),
                }

                torch.save(content, os.path.join(exp_path, "content.pth"))

        if epoch % args.save_ckpt_every == 0:
            if args.use_ema:
                optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

            torch.save(
                netG.state_dict(),
                os.path.join(exp_path, "netG_{}.pth".format(epoch)),
            )
            if args.use_ema:
                optimizerG.swap_parameters_with_ema(store_params_in_ema=True)


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

    # geenrator and training
    parser.add_argument(
        "--exp", default="experiment_cifar_default", help="name of experiment"
    )
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
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
    parser.add_argument("--ssim", action="store_true", default=False)

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
    train(args)
