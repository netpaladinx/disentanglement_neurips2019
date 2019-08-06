'''
WGAN-GP: score_z_true: 3~, score_z_pred: 1.5~
1. discriminator with high complexity
2. more inner loop to train discriminator with smaller lr and zero beta1
3. train encoder with relatively larger lr and also zero beta1
'''
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np


def scatter_dim_pairs(x, name, idx_pairs=None):
    n_latent = x.size(1)
    fig = plt.figure(figsize=(20, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        latent_1 = np.random.randint(n_latent) if idx_pairs is None else idx_pairs[i][0]
        latent_2 = np.random.randint(n_latent) if idx_pairs is None else idx_pairs[i][1]
        points_x = x[:, latent_1]
        points_y = x[:, latent_2]
        plt.scatter(points_x.numpy(), points_y.numpy(), s=10)
        plt.xlabel('[%s] dim_1: %d, dim_2: %d' % (name, latent_1, latent_2), fontsize=10)
    plt.show()
    fig.savefig('%s.pdf' % name)


n_latent = 10
n_samples = 500
z_guassian = torch.randn(n_samples, n_latent).cuda()
#scatter_dim_pairs(z_guassian.cpu(), 'z_guassian')


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_latent, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 100), nn.Tanh()
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nr, nc = m.weight.size()
                scale1 = np.random.rand() * 10
                offset1 = np.random.rand() * 20
                scale2 = np.random.rand() * 30
                offset2 = np.random.rand() * 40
                w = (torch.arange(0, nr).unsqueeze(1) * torch.arange(0, nc).unsqueeze(0)).to(torch.float) / (nr * nc)
                w = (torch.sin(scale1 * w + offset1) + torch.cos(scale2 * w + offset2)) * 0.1
                w2 = w.clone()
                w[torch.randperm(w.size(0))] = w
                w[:, torch.randperm(w.size(1))] = w
                w = (w * w2).abs().pow(0.5) * w2.sign()
                m.weight.data.copy_(w)

        self.backbone.apply(init_weights)

    def forward(self, x):
        return self.backbone(x)


idx_pairs = []
for _ in range(5):
    for _ in range(5):
        idx1 = np.random.randint(n_latent)
        idx2 = np.random.randint(n_latent)
        idx_pairs.append((idx1, idx2))

generator = Generator().cuda()
x_samples = generator(z_guassian).detach()
scatter_dim_pairs(x_samples.cpu(), 'x_samples', idx_pairs=idx_pairs)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(100, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, n_latent)
        )

    def forward(self, x):
        return self.backbone(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_latent, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 100), nn.Tanh()
        )

    def forward(self, x):
        return self.backbone(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_latent, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def loss_from_guassian(x_gen):
    ''' x_gen: batch_size x n_dims
    '''
    x_mean = torch.mean(x_gen, 0)
    x_center = x_gen - x_mean
    x_cov = x_center.t().mm(x_center) / x_gen.size(0)
    x_cov_diag = x_cov.diag()
    x_cov_offdiag = x_cov - x_cov_diag.diag()

    corr_x_mean = (x_mean - 0).pow(2).sum()
    corr_x_cov_diag = (x_cov_diag - 1).pow(2).sum()
    corr_x_cov_offdiag = (x_cov_offdiag - 0).pow(2).sum()
    return corr_x_mean + corr_x_cov_diag + 100 * corr_x_cov_offdiag


encoder = Encoder().cuda()
decoder = Decoder().cuda()
discrminator = Discriminator().cuda()

vae_optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=0.0005, betas=(0, 0.9))
discr_optimizer = optim.Adam(discrminator.parameters(), lr=0.0001, betas=(0, 0.9))

batch_size = 128
for i in range(1000):
    for j in range(5):
        z_true = torch.randn(batch_size, n_latent).cuda()
        z_latent = torch.randn(batch_size, n_latent).cuda()
        x_samples = generator(z_latent).detach()

        z_pred = encoder(x_samples).detach()
        t = torch.rand(z_pred.size(0), 1).cuda()
        z_inter = t * z_pred + (1 - t) * z_true
        z_inter.requires_grad_()

        score_z_true = discrminator(z_true)
        score_z_pred = discrminator(z_pred)
        score_z_inter = discrminator(z_inter)
        score_z_inter.backward(gradient=torch.ones_like(score_z_inter).cuda(), create_graph=True)
        gradient_penalty = (torch.norm(z_inter.grad, 'fro', 1) - 1).pow(2).mean()
        discr_loss = (score_z_pred - score_z_true).mean() + 10 * gradient_penalty
        discr_optimizer.zero_grad()
        discr_loss.backward()
        discr_optimizer.step()

        if i % 100 == 0:
            print('step %d - %d' % (i, j))
            print('score_z_true: ', score_z_true.detach().mean().cpu().item())
            print('score_z_pred: ', score_z_pred.detach().mean().cpu().item())
            print('score_z_inter: ', score_z_inter.detach().mean().cpu().item())

    for j in range(1):
        z_latent = torch.randn(batch_size, n_latent).cuda()
        x_samples = generator(z_latent).detach()

        z_pred = encoder(x_samples)
        neg_score_z_pred = - discrminator(z_pred).mean()
        recon_loss = F.mse_loss(decoder(z_pred), x_samples)
        vae_loss = recon_loss + neg_score_z_pred
        vae_optimizer.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()

        if i % 100 == 0:
            print('step %d - %d' % (i, j))
            print('neg_score_z_pred: ', neg_score_z_pred.detach().cpu().item())
            print('recon_loss: ', recon_loss.detach().cpu().item())

    #scatter_dim_pairs(z_pred.detach().cpu(), 'z_pred')

x_samples = generator(z_guassian).detach()
z_pred = encoder(x_samples)
x_pred = decoder(z_pred)
scatter_dim_pairs(z_pred.detach().cpu(), 'z_pred')
scatter_dim_pairs(x_samples.detach().cpu(), 'x_samples', idx_pairs=idx_pairs)
scatter_dim_pairs(x_pred.detach().cpu(), 'x_pred', idx_pairs=idx_pairs)