import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, kl

from .unet import Unet
from .utils import init_weights, init_weights_orthogonal_normal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_channels, num_filters, no_convs_per_block, padding=True, posterior=False):
        super().__init__()
        self.input_channels = input_channels + 1 if posterior else input_channels
        self.num_filters = num_filters

        layers = []
        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, inputs):
        return self.layers(inputs)


class AxisAlignedConvGaussian(nn.Module):
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, posterior=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(
            input_channels=input_channels,
            num_filters=num_filters,
            no_convs_per_block=no_convs_per_block,
            posterior=posterior,
        )
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1, 1), stride=1)
        nn.init.kaiming_normal_(self.conv_layer.weight, mode="fan_in", nonlinearity="relu")
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, inputs, segm=None):
        if segm is not None:
            inputs = torch.cat((inputs, segm), dim=1)

        encoding = self.encoder(inputs)
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        mu_log_sigma = self.conv_layer(encoding).squeeze(dim=2).squeeze(dim=2)
        mu = mu_log_sigma[:, : self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim :]
        return Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)


class Fcomb(nn.Module):
    def __init__(self, num_filters, latent_dim, num_classes, no_convs_fcomb, use_tile=True):
        super().__init__()
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile

        layers = []
        layers.append(nn.Conv2d(self.num_filters[0] + self.latent_dim, self.num_filters[0], kernel_size=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(no_convs_fcomb - 2):
            layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)
        self.layers.apply(init_weights_orthogonal_normal)
        self.last_layer.apply(init_weights_orthogonal_normal)

    def tile(self, tensor, dim, n_tile):
        init_dim = tensor.size(dim)
        repeat_idx = [1] * tensor.dim()
        repeat_idx[dim] = n_tile
        tensor = tensor.repeat(*repeat_idx)
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
        ).to(device)
        return torch.index_select(tensor, dim, order_index)

    def forward(self, feature_map, z):
        z = torch.unsqueeze(z, 2)
        z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
        z = torch.unsqueeze(z, 3)
        z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])
        feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
        output = self.layers(feature_map)
        return self.last_layer(output)


class ProbabilisticUnet(nn.Module):
    def __init__(
        self,
        input_channels=1,
        num_classes=1,
        num_filters=None,
        latent_dim=6,
        no_convs_fcomb=4,
        beta=10.0,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters or [32, 64, 128, 192]
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.beta = beta
        self.z_prior_sample = 0

        self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, apply_last_layer=False, padding=True).to(device)
        self.prior = AxisAlignedConvGaussian(
            self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim
        ).to(device)
        self.posterior = AxisAlignedConvGaussian(
            self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, posterior=True
        ).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.num_classes, self.no_convs_fcomb).to(device)

    def forward(self, patch, segm, training=True):
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch)

    def sample(self, testing=False):
        z_prior = self.prior_latent_space.sample() if testing else self.prior_latent_space.rsample()
        self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features, z_prior)

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        elif calculate_posterior:
            z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        if analytic:
            return kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        if calculate_posterior:
            z_posterior = self.posterior_latent_space.rsample()
        log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
        log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
        return log_posterior_prob - log_prior_prob

    def elbo(self, segm, analytic_kl=True, reconstruct_posterior_mean=False):
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        z_posterior = self.posterior_latent_space.rsample()
        self.kl = torch.mean(
            self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior)
        )
        self.reconstruction = self.reconstruct(
            use_posterior_mean=reconstruct_posterior_mean,
            calculate_posterior=False,
            z_posterior=z_posterior,
        )
        reconstruction_loss = criterion(input=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)
        return -(self.reconstruction_loss + self.beta * self.kl)

