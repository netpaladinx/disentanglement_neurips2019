from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def scatter_dim_pairs(x, name, nr=5, nc=5, idx_pairs=None):
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        x1, x2 = x[0], x[1]
    else:
        x1, x2 = x, x
    n_dims_1 = x1.size(1)
    n_dims_2 = x2.size(1)
    fig = plt.figure(figsize=(20, 10))
    for i in range(nr * nc):
        plt.subplot(nr, nc, i+1)
        dim_1 = np.random.randint(n_dims_1) if idx_pairs is None else idx_pairs[i][0]
        dim_2 = np.random.randint(n_dims_2) if idx_pairs is None else idx_pairs[i][1]
        points_x = x1[:, dim_1]
        points_y = x2[:, dim_2]
        plt.scatter(points_x.numpy(), points_y.numpy(), s=10)
        plt.xlabel('[%s] dim_1: %d, dim_2: %d' % (name, dim_1, dim_2), fontsize=10)
    plt.show(block=False)
    fig.savefig('%s.pdf' % name)


def tsne(x, name):
    x_2d = TSNE(n_components=2).fit_transform(x.numpy())
    fig = plt.figure()
    plt.scatter(x_2d[:, 0], x_2d[:, 1], s=10)
    plt.xlabel('%s' % name, fontsize=10)
    plt.show(block=False)
    fig.savefig('%s.pdf' % name)


n_latent = 10
n_discrete = 4
n_dims = 100

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_latent * 2, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 100), nn.Tanh()
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                weight = torch.empty_like(m.weight)
                nr, nc = m.weight.size()
                T = np.pi * 2
                scale1 = np.random.rand() * 10
                offset1 = np.random.rand()
                scale2 = np.random.rand() * 10
                offset2 = np.random.rand()
                xy = (torch.arange(0, nr).unsqueeze(1) * torch.arange(0, nc).unsqueeze(0)).to(torch.float) / (nr * nc)
                w = (torch.sin(scale1 * T * xy + offset1 * T) + torch.cos(scale2 * T * xy + offset2 * T)) * 0.05
                w2 = w[torch.randperm(nr)]
                w3 = w2[:, torch.randperm(nc)]
                weight = ((w * w2 + w * w3) / 2).abs().pow(1/2) * w.sign()
                m.weight.data.copy_(weight)

        self.backbone.apply(init_weights)

        self.embedding = nn.Parameter(torch.rand(n_discrete, n_latent) * 2 - 1)

    def forward(self, z_guassian, z_discrete):
        z_embedding = self.embedding.index_select(0, z_discrete)
        z = torch.cat((z_guassian, z_embedding), 1)
        return self.backbone(z), z_embedding


batch_size = 256
group_size = 32
my_z_guassian = torch.randn(batch_size, n_latent).cuda()
my_z_discrete = torch.randint(n_discrete, (batch_size,)).cuda()
generator = Generator().cuda()
my_x_samples, my_z_embedding = generator(my_z_guassian, my_z_discrete)

dim_idx_pairs = [(0, 0), (0, 1), (0, 5), (0, 10), (0, 50),
                 (8, 2), (8, 8), (8, 23), (8, 65), (8, 92),
                 (26, 6), (26, 17), (26, 26), (26, 72), (26, 88),
                 (56, 9), (56, 32), (56, 41), (56, 56), (56, 95),
                 (86, 11), (86, 33), (86, 55), (86, 66), (86, 86)]
scatter_dim_pairs(my_x_samples.detach().cpu(), 'my_x_samples', nr=5, nc=5, idx_pairs=dim_idx_pairs)
tsne(my_x_samples.detach().cpu(), 'my_x_samples_tsne')
tsne(my_z_embedding.detach().cpu(), 'my_z_embedding_tsne')
tsne(my_z_guassian.detach().cpu(), 'my_z_guassian_tsne')


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_dims, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
        )
        self.fc_continous = nn.Linear(1024, n_latent)
        self.fc_discrete = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.backbone(x)
        z_continous = self.fc_continous(x)
        z_discrete_p = F.softmax(self.fc_discrete(x), 1)
        discrete_idx = torch.argmax(z_discrete_p, 1, True)
        mask = torch.zeros_like(z_discrete_p).scatter_(1, discrete_idx, torch.ones_like(discrete_idx).float())
        scale = mask / (z_discrete_p.detach() * mask).sum(1, True)
        z_discrete = z_discrete_p * scale
        return z_continous, z_discrete, z_discrete_p


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_latent * 2, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, n_dims), nn.Tanh()
        )
        self.embedding = nn.Parameter(torch.randn(n_discrete, n_latent))

    def forward(self, z_cont, z_disc):
        z_embed = z_disc.mm(self.embedding)
        z = torch.cat((z_cont, z_embed), 1)
        return self.backbone(z)


class DiscriminatorCont(nn.Module):
    def __init__(self):
        super(DiscriminatorCont, self).__init__()
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


class DiscriminatorDisc(nn.Module):
    def __init__(self):
        super(DiscriminatorDisc, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_discrete * group_size, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


encoder = Encoder().cuda()
decoder = Decoder().cuda()
discr_cont = DiscriminatorCont().cuda()
discr_disc = DiscriminatorDisc().cuda()

encdec_optimzier = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=0.0005, betas=(0, 0.9))
discr_cont_optimizer = optim.Adam(discr_cont.parameters(), lr=0.0001, betas=(0, 0.9))
discr_disc_optimizer = optim.Adam(discr_disc.parameters(), lr=0.0001, betas=(0, 0.9))

for step in range(1000):

    score_z_cont_metrics = []
    score_z_cont_tar_metrics = []
    score_z_disc_metrics = []
    score_z_disc_tar_metrics = []

    for k in range(5):
        z_guassian = torch.randn(batch_size, n_latent).cuda()
        z_discrete = torch.randint(n_discrete, (batch_size,)).cuda()
        x_samples, _ = generator(z_guassian, z_discrete)
        z_cont, z_disc, z_disc_p = encoder(x_samples.detach())
        z_disc_grp = z_disc.reshape(-1, group_size * z_disc.size(1))

        z_cont_tar = torch.randn_like(z_guassian)
        z_disc_tar = F.one_hot(torch.randint_like(z_discrete, n_discrete), n_discrete).float()
        z_disc_tar_grp = z_disc_tar.reshape(-1, group_size * z_disc_tar.size(1))

        # for z_cont
        z_cont = z_cont.detach()
        score_z_cont = discr_cont(z_cont).mean()
        score_z_cont_tar = discr_cont(z_cont_tar).mean()
        discr_cont_loss = score_z_cont - score_z_cont_tar
        score_z_cont_metrics.append(score_z_cont.detach().cpu().item())
        score_z_cont_tar_metrics.append(score_z_cont_tar.detach().cpu().item())

        t = torch.rand(batch_size, 1).cuda()
        z_cont_inter = t * z_cont + (1 - t) * z_cont_tar
        z_cont_inter.requires_grad_()
        score_z_cont_inter = discr_cont(z_cont_inter)
        score_z_cont_inter.backward(gradient=torch.ones_like(score_z_cont_inter), create_graph=True)
        gradient_penalty = (torch.norm(z_cont_inter.grad, 'fro', 1) - 1).pow(2).mean()
        discr_cont_loss = discr_cont_loss + 10 * gradient_penalty

        discr_cont_optimizer.zero_grad()
        discr_cont_loss.backward()
        discr_cont_optimizer.step()

        # for z_disc
        z_disc_grp = z_disc_grp.detach()
        score_z_disc = discr_disc(z_disc_grp).mean()
        score_z_disc_tar = discr_disc(z_disc_tar_grp).mean()
        discr_disc_loss = score_z_disc - score_z_disc_tar
        score_z_disc_metrics.append(score_z_disc.detach().cpu().item())
        score_z_disc_tar_metrics.append(score_z_disc_tar.detach().cpu().item())

        t = torch.rand(batch_size // group_size, 1).cuda()
        z_disc_inter = t * z_disc_grp + (1 - t) * z_disc_tar_grp
        z_disc_inter.requires_grad_()
        score_z_disc_inter = discr_disc(z_disc_inter)
        score_z_disc_inter.backward(gradient=torch.ones_like(score_z_disc_inter), create_graph=True)
        gradient_penalty = (torch.norm(z_disc_inter.grad, 'fro', 1) - 1).pow(2).mean()
        discr_disc_loss = discr_disc_loss + 10 * gradient_penalty

        discr_disc_optimizer.zero_grad()
        discr_disc_loss.backward()
        discr_disc_optimizer.step()

    if step % 100 == 0:
        print('Step %d [Training Discriminator]' % step)
        print('  score_z_cont: %s' % ', '.join(map(lambda x: str(x), score_z_cont_metrics)))
        print('  score_z_cont_tar: %s' % ', '.join(map(lambda x: str(x), score_z_cont_tar_metrics)))
        print('  score_z_disc: %s' % ', '.join(map(lambda x: str(x), score_z_disc_metrics)))
        print('  score_z_disc_tar: %s' % ', '.join(map(lambda x: str(x), score_z_disc_tar_metrics)))

    score_z_cont_metrics = []
    score_z_disc_metrics = []

    for k in range(1):
        z_guassian = torch.randn(batch_size, n_latent).cuda()
        z_discrete = torch.randint(n_discrete, (batch_size,)).cuda()
        x_samples, _ = generator(z_guassian, z_discrete)
        z_cont, z_disc, z_disc_p = encoder(x_samples.detach())
        z_disc_grp = z_disc.reshape(-1, group_size * z_disc.size(1))

        score_z_cont = discr_cont(z_cont).mean()
        score_z_disc = discr_disc(z_disc_grp).mean()
        score_z_cont_metrics.append(score_z_cont.detach().cpu().item())
        score_z_disc_metrics.append(score_z_disc.detach().cpu().item())

        x_pred = decoder(z_cont, z_disc)
        recon_loss = F.mse_loss(x_pred, x_samples)
        encdec_loss = recon_loss - score_z_cont - score_z_disc

        encdec_optimzier.zero_grad()
        encdec_loss.backward()
        encdec_optimzier.step()

    if step % 100 == 0:
        print('Step %d [Training Encoder & Decoder]' % step)
        print('  score_z_cont: %s' % ', '.join(map(lambda x: str(x), score_z_cont_metrics)))
        print('  score_z_disc: %s' % ', '.join(map(lambda x: str(x), score_z_disc_metrics)))


my_z_cont, my_z_disc, my_z_disc_p = encoder(my_x_samples)
my_x_pred = decoder(my_z_cont, my_z_disc)

scatter_dim_pairs(my_x_pred.detach().cpu(), 'my_x_pred', nr=5, nc=5, idx_pairs=dim_idx_pairs)
tsne(my_x_pred.detach().cpu(), 'my_x_pred_tsne')

latent_idx_pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                    (5, 6), (5, 7), (5, 8), (5, 9), (5, 4)]
scatter_dim_pairs(my_z_cont.detach().cpu(), 'my_z_cont', nr=3, nc=5, idx_pairs=latent_idx_pairs)
tsne(my_z_cont.detach().cpu(), 'my_z_cont_tsne')

discrete_idx_pairs = [(0, 0), (0, 1), (0, 2), (0, 3),
                      (1, 0), (1, 1), (1, 2), (1, 3),
                      (2, 0), (2, 1), (2, 2), (2, 3),
                      (3, 0), (3, 1), (3, 2), (3, 3)]
my_z_discrete = F.one_hot(my_z_discrete, n_discrete).float()
scatter_dim_pairs((my_z_discrete.detach().cpu(), my_z_disc.detach().cpu()), 'corr_z_discrete', nr=4, nc=4, idx_pairs=discrete_idx_pairs)
tsne(my_z_disc_p.detach().cpu(), 'my_z_disc_p_tsne')
print('my_z_disc_p', my_z_disc_p.detach().sum(0).cpu())
print('my_z_disc', my_z_disc.detach().sum(0).cpu())
print('my_z_discrete', my_z_discrete.detach().sum(0).cpu())