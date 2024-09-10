

import torch
import numpy as np

import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10



BETA_MIN = .05
BETA_MAX = .2
USE_GEOMETRIC = False


# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(device):
    n_timestep = 4
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule( device):
    n_timestep = 4
    beta_min = BETA_MIN
    beta_max = BETA_MAX
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if USE_GEOMETRIC:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients():
    def __init__(self,device):
        self.sigmas, self.a_s, _ = get_sigma_schedule( device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
        self.t = 1


    def q_sample(self, x_start,  *, noise=None):
        """
        Diffuse the data (t == 0 means diffused for t step)
        """
        time = torch.full((x_start.size(0),), self.t, dtype=torch.int64).to(x_start.device)
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = extract(self.a_s_cum, time, x_start.shape) * x_start + \
              extract(self.sigmas_cum, time, x_start.shape) * noise

        return x_t


    def q_sample_pairs(self, x_start):
        """
        Generate a pair of disturbed images for training
        :param x_start: x_0
        :param t: time step t
        :return: x_t, x_{t+1}
        """
        time = torch.full((x_start.size(0),), self.t, dtype=torch.int64).to(x_start.device)

        noise = torch.randn_like(x_start)
        x_t = self.q_sample( x_start)
        x_t_plus_one = extract(self.a_s, time + 1, x_start.shape) * x_t + \
                       extract(self.sigmas, time + 1, x_start.shape) * noise

        return x_t, x_t_plus_one


# %% posterior sampling
class Posterior_Coefficients():
    def __init__(self, device):
        _, _, self.betas = get_sigma_schedule( device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                    (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.t = 1


    def sample_posterior(self, x_0, x_t):
        def q_posterior(x_0, x_t):
            time = torch.full((x_0.size(0),), self.t, dtype=torch.int64).to(x_0.device)

            mean = (
                    extract(self.posterior_mean_coef1, time, x_t.shape) * x_0
                    + extract(self.posterior_mean_coef2, time, x_t.shape) * x_t
            )
            var = extract(self.posterior_variance, time, x_t.shape)
            log_var_clipped = extract(self.posterior_log_variance_clipped, time, x_t.shape)
            return mean, var, log_var_clipped

        def p_sample(x_0, x_t):
            time = torch.full((x_0.size(0),), self.t, dtype=torch.int64).to(x_0.device)
            mean, _, log_var = q_posterior(x_0, x_t)

            noise = torch.randn_like(x_t)

            nonzero_mask = (1 - (time == 0).type(torch.float32))

            return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

        sample_x_pos = p_sample(x_0, x_t,)

        return sample_x_pos



