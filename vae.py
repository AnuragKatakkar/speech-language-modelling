"""
This file defines the various VAE models.
"""

import torch
import torch.nn as nn

from models import ConvEncoder, ConvDecoder

class VAE(nn.Module):
    def __init__(self, encoder_type="Conv", decoder_type="Conv", encoder_dims=None, decoder_dims=None, embedding_dim=512):
        super(VAE, self).__init__()

        self.embedding_dim = embedding_dim

        if encoder_type=="Conv":
            self.encoder = ConvEncoder(encoder_dims, embedding_dim)

        if decoder_type=="Conv":
            self.decoder = ConvDecoder(decoder_dims, embedding_dim)

        self.fc_mu = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.fc_var = nn.Linear(self.embedding_dim, self.embedding_dim)

    def encode(self, x):
        latent = self.encoder(x)
        return self.fc_mu(latent), self.fc_var(latent)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(logvar)
        return eps*std + mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), x, mu, logvar

    def sample(self, num_samples, device="cuda"):
        z = torch.randn(num_samples, self.embedding_dim).to(device)
        return self.decode(z)

    def latent_loss(self, mean, stddev):
        mean_sq = mean*mean
        var = stddev*stddev
        eps = 1e-10
        return 0.5 * torch.mean(mean_sq + var - torch.log(var + eps) - 1)


def _make_vae():
    """
        Returns the desired VAE model. Note that we currently only support fully
        convolutional VAEs.
    """
    hidden_channels = [32, 64, 128, 512, 512]
    embedding_dim = 512
    return VAE(encoder_dims=hidden_channels, decoder_dims=hidden_channels, embedding_dim=512,
                encoder_type="Conv", decoder_type="Conv")
