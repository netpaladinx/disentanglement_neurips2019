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
    x_2d = TSNE(n_components=2).fit_transform(x.numpy() if isinstance(x, torch.Tensor) else x)
    fig = plt.figure()
    plt.scatter(x_2d[:, 0], x_2d[:, 1], s=10)
    plt.xlabel('%s' % name, fontsize=10)
    plt.show(block=False)
    fig.savefig('%s.pdf' % name)


n_latent = 10
n_discrete_1 = 3
n_discrete_2 = 4
n_discrete_3 = 5
n_dims = 100
prob_discrete_1 = [0.1, 0.3, 0.6]
prob_discrete_2 = [0.1, 0.2, 0.3, 0.4]
prob_discrete_3 = [0.05, 0.1, 0.25, 0.3, 0.3]


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_latent * 4, 200), nn.ReLU(),
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

        self.embedding_1 = nn.Parameter(torch.rand(n_discrete_1, n_latent) * 2 - 1)
        self.embedding_2 = nn.Parameter(torch.rand(n_discrete_2, n_latent) * 2 - 1)
        self.embedding_3 = nn.Parameter(torch.rand(n_discrete_3, n_latent) * 2 - 1)

    def forward(self, z_guassian, z_discrete_1, z_discrete_2, z_discrete_3):
        z_embedding_1 = self.embedding_1.index_select(0, z_discrete_1)
        z_embedding_2 = self.embedding_2.index_select(0, z_discrete_2)
        z_embedding_3 = self.embedding_3.index_select(0, z_discrete_3)
        z = torch.cat((z_guassian, z_embedding_1, z_embedding_2, z_embedding_3), 1)
        return self.backbone(z)


batch_size = 256
group_size = 32
my_z_guassian = torch.randn(batch_size, n_latent).cuda()

my_z_discrete_1 = torch.from_numpy(np.random.choice(n_discrete_1, batch_size, p=prob_discrete_1)).cuda()
my_z_discrete_2 = torch.from_numpy(np.random.choice(n_discrete_2, batch_size, p=prob_discrete_2)).cuda()
my_z_discrete_3 = torch.from_numpy(np.random.choice(n_discrete_3, batch_size, p=prob_discrete_3)).cuda()
generator = Generator().cuda()
my_x_samples = generator(my_z_guassian, my_z_discrete_1, my_z_discrete_2, my_z_discrete_3)

dim_idx_pairs = [(0, 0), (0, 1), (0, 5), (0, 10), (0, 50),
                 (8, 2), (8, 8), (8, 23), (8, 65), (8, 92),
                 (26, 6), (26, 17), (26, 26), (26, 72), (26, 88),
                 (56, 9), (56, 32), (56, 41), (56, 56), (56, 95),
                 (86, 11), (86, 33), (86, 55), (86, 66), (86, 86)]
scatter_dim_pairs(my_x_samples.detach().cpu(), 'my_x_samples', nr=5, nc=5, idx_pairs=dim_idx_pairs)
tsne(my_x_samples.detach().cpu(), 'my_x_samples_tsne')
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
        self.fc_discrete_1 = nn.Linear(1024, n_discrete_1)
        self.fc_discrete_2 = nn.Linear(1024, n_discrete_2)
        self.fc_discrete_3 = nn.Linear(1024, n_discrete_3)

        self.z_discrete_p = []

    def forward(self, x):
        x = self.backbone(x)
        z_continous = self.fc_continous(x)
        z_discrete_p1 = F.softmax(self.fc_discrete_1(x), 1)
        z_discrete_p2 = F.softmax(self.fc_discrete_2(x), 1)
        z_discrete_p3 = F.softmax(self.fc_discrete_3(x), 1)

        self.z_discrete_p = [z_discrete_p1.detach().cpu().numpy(),
                             z_discrete_p2.detach().cpu().numpy(),
                             z_discrete_p3.detach().cpu().numpy()]

        discrete_idx1 = torch.argmax(z_discrete_p1, 1, True)
        discrete_idx2 = torch.argmax(z_discrete_p2, 1, True)
        discrete_idx3 = torch.argmax(z_discrete_p3, 1, True)
        mask1 = torch.zeros_like(z_discrete_p1).scatter_(1, discrete_idx1, torch.ones_like(discrete_idx1).float())
        scale1 = mask1 / (z_discrete_p1.detach() * mask1).sum(1, True)
        mask2 = torch.zeros_like(z_discrete_p2).scatter_(1, discrete_idx2, torch.ones_like(discrete_idx2).float())
        scale2 = mask2 / (z_discrete_p2.detach() * mask2).sum(1, True)
        mask3 = torch.zeros_like(z_discrete_p3).scatter_(1, discrete_idx3, torch.ones_like(discrete_idx3).float())
        scale3 = mask3 / (z_discrete_p3.detach() * mask3).sum(1, True)
        z_discrete1 = z_discrete_p1 * scale1
        z_discrete2 = z_discrete_p2 * scale2
        z_discrete3 = z_discrete_p3 * scale3
        return z_continous, z_discrete1, z_discrete2, z_discrete3


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_latent * 4, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, n_dims), nn.Tanh()
        )
        self.embedding_1 = nn.Parameter(torch.randn(n_discrete_1, n_latent))
        self.embedding_2 = nn.Parameter(torch.randn(n_discrete_2, n_latent))
        self.embedding_3 = nn.Parameter(torch.randn(n_discrete_3, n_latent))

    def forward(self, z_cont, z_disc1, z_disc2, z_disc3):
        z_embed1 = z_disc1.mm(self.embedding_1)
        z_embed2 = z_disc2.mm(self.embedding_2)
        z_embed3 = z_disc3.mm(self.embedding_3)
        z = torch.cat((z_cont, z_embed1, z_embed2, z_embed3), 1)
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
    def __init__(self, n_dims):
        super(DiscriminatorDisc, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_dims, 1024), nn.LeakyReLU(),
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
discr_disc1 = DiscriminatorDisc(n_discrete_1 * group_size).cuda()
discr_disc2 = DiscriminatorDisc(n_discrete_2 * group_size).cuda()
discr_disc3 = DiscriminatorDisc(n_discrete_3 * group_size).cuda()
discr_disc = DiscriminatorDisc((n_discrete_1 + n_discrete_2 + n_discrete_3) * group_size // 2).cuda()

encdec_optimzier = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=0.0005, betas=(0, 0.9))
discr_cont_optimizer = optim.Adam(discr_cont.parameters(), lr=0.0001, betas=(0, 0.9))
discr_disc_optimizer = optim.Adam(chain(discr_disc1.parameters(),
                                        discr_disc2.parameters(),
                                        discr_disc2.parameters(),
                                        discr_disc.parameters()), lr=0.0001, betas=(0, 0.9))

for step in range(10000):

    score_z_cont_metrics = []
    score_z_cont_tar_metrics = []
    score_z_disc_metrics = []
    score_z_disc_tar_metrics = []
    score_z_disc1_metrics = []
    score_z_disc_tar1_metrics = []
    score_z_disc2_metrics = []
    score_z_disc_tar2_metrics = []
    score_z_disc3_metrics = []
    score_z_disc_tar3_metrics = []

    for k in range(5):
        z_guassian = torch.randn(batch_size, n_latent).cuda()
        z_discrete_1 = torch.from_numpy(np.random.choice(n_discrete_1, batch_size, p=prob_discrete_1)).cuda()
        z_discrete_2 = torch.from_numpy(np.random.choice(n_discrete_2, batch_size, p=prob_discrete_2)).cuda()
        z_discrete_3 = torch.from_numpy(np.random.choice(n_discrete_3, batch_size, p=prob_discrete_3)).cuda()
        x_samples = generator(z_guassian, z_discrete_1, z_discrete_2, z_discrete_3)
        z_cont, z_disc1, z_disc2, z_disc3 = encoder(x_samples.detach())
        z_disc = torch.cat((z_disc1, z_disc2, z_disc3), 1)
        z_disc1_grp = z_disc1.reshape(-1, group_size * z_disc1.size(1))
        z_disc2_grp = z_disc2.reshape(-1, group_size * z_disc2.size(1))
        z_disc3_grp = z_disc3.reshape(-1, group_size * z_disc3.size(1))
        z_disc_grp = z_disc.reshape(-1, group_size // 2 * z_disc.size(1))

        z_cont_tar = torch.randn_like(z_guassian)
        z_disc_tar1 = F.one_hot(torch.from_numpy(np.random.choice(n_discrete_1, batch_size, p=prob_discrete_1)).cuda()).float()
        z_disc_tar2 = F.one_hot(torch.from_numpy(np.random.choice(n_discrete_2, batch_size, p=prob_discrete_2)).cuda()).float()
        z_disc_tar3 = F.one_hot(torch.from_numpy(np.random.choice(n_discrete_3, batch_size, p=prob_discrete_3)).cuda()).float()
        z_disc_tar = torch.cat((z_disc_tar1, z_disc_tar2, z_disc_tar3), 1)
        z_disc_tar1_grp = z_disc_tar1.reshape(-1, group_size * z_disc_tar1.size(1))
        z_disc_tar2_grp = z_disc_tar2.reshape(-1, group_size * z_disc_tar2.size(1))
        z_disc_tar3_grp = z_disc_tar3.reshape(-1, group_size * z_disc_tar3.size(1))
        z_disc_tar_grp = z_disc_tar.reshape(-1, group_size // 2 * z_disc_tar.size(1))

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
        z_disc1_grp = z_disc1_grp.detach()
        score_z_disc1 = discr_disc1(z_disc1_grp).mean()
        score_z_disc_tar1 = discr_disc1(z_disc_tar1_grp).mean()
        z_disc2_grp = z_disc2_grp.detach()
        score_z_disc2 = discr_disc2(z_disc2_grp).mean()
        score_z_disc_tar2 = discr_disc2(z_disc_tar2_grp).mean()
        z_disc3_grp = z_disc3_grp.detach()
        score_z_disc3 = discr_disc3(z_disc3_grp).mean()
        score_z_disc_tar3 = discr_disc3(z_disc_tar3_grp).mean()
        z_disc_grp = z_disc_grp.detach()
        score_z_disc = discr_disc(z_disc_grp).mean()
        score_z_disc_tar = discr_disc(z_disc_tar_grp).mean()

        discr_disc_loss = (score_z_disc1 - score_z_disc_tar1 + score_z_disc2 - score_z_disc_tar2 +
                           score_z_disc3 - score_z_disc_tar3 + score_z_disc - score_z_disc_tar)
        score_z_disc1_metrics.append(score_z_disc1.detach().cpu().item())
        score_z_disc_tar1_metrics.append(score_z_disc_tar1.detach().cpu().item())
        score_z_disc2_metrics.append(score_z_disc2.detach().cpu().item())
        score_z_disc_tar2_metrics.append(score_z_disc_tar2.detach().cpu().item())
        score_z_disc3_metrics.append(score_z_disc3.detach().cpu().item())
        score_z_disc_tar3_metrics.append(score_z_disc_tar3.detach().cpu().item())
        score_z_disc_metrics.append(score_z_disc.detach().cpu().item())
        score_z_disc_tar_metrics.append(score_z_disc_tar.detach().cpu().item())

        t = torch.rand(batch_size // group_size, 1).cuda()
        z_disc_inter1 = t * z_disc1_grp + (1 - t) * z_disc_tar1_grp
        z_disc_inter1.requires_grad_()
        score_z_disc_inter = discr_disc1(z_disc_inter1)
        score_z_disc_inter.backward(gradient=torch.ones_like(score_z_disc_inter), create_graph=True)
        gradient_penalty1 = (torch.norm(z_disc_inter1.grad, 'fro', 1) - 1).pow(2).mean()

        t = torch.rand(batch_size // group_size, 1).cuda()
        z_disc_inter2 = t * z_disc2_grp + (1 - t) * z_disc_tar2_grp
        z_disc_inter2.requires_grad_()
        score_z_disc_inter = discr_disc2(z_disc_inter2)
        score_z_disc_inter.backward(gradient=torch.ones_like(score_z_disc_inter), create_graph=True)
        gradient_penalty2 = (torch.norm(z_disc_inter2.grad, 'fro', 1) - 1).pow(2).mean()

        t = torch.rand(batch_size // group_size, 1).cuda()
        z_disc_inter3 = t * z_disc3_grp + (1 - t) * z_disc_tar3_grp
        z_disc_inter3.requires_grad_()
        score_z_disc_inter = discr_disc3(z_disc_inter3)
        score_z_disc_inter.backward(gradient=torch.ones_like(score_z_disc_inter), create_graph=True)
        gradient_penalty3 = (torch.norm(z_disc_inter3.grad, 'fro', 1) - 1).pow(2).mean()

        t = torch.rand(batch_size // group_size * 2, 1).cuda()
        z_disc_inter = t * z_disc_grp + (1 - t) * z_disc_tar_grp
        z_disc_inter.requires_grad_()
        score_z_disc_inter = discr_disc(z_disc_inter)
        score_z_disc_inter.backward(gradient=torch.ones_like(score_z_disc_inter), create_graph=True)
        gradient_penalty = (torch.norm(z_disc_inter.grad, 'fro', 1) - 1).pow(2).mean()

        discr_disc_loss = discr_disc_loss + 10 * (gradient_penalty1 + gradient_penalty2 + gradient_penalty3 + gradient_penalty)

        discr_disc_optimizer.zero_grad()
        discr_disc_loss.backward()
        discr_disc_optimizer.step()

    if step % 100 == 0:
        print('Step %d [Training Discriminator]' % step)
        print('  score_z_cont: %s' % ', '.join(map(lambda x: str(x), score_z_cont_metrics)))
        print('  score_z_cont_tar: %s' % ', '.join(map(lambda x: str(x), score_z_cont_tar_metrics)))
        print('  score_z_disc: %s' % ', '.join(map(lambda x: str(x), score_z_disc_metrics)))
        print('  score_z_disc_tar: %s' % ', '.join(map(lambda x: str(x), score_z_disc_tar_metrics)))
        print('  score_z_disc1: %s' % ', '.join(map(lambda x: str(x), score_z_disc1_metrics)))
        print('  score_z_disc_tar1: %s' % ', '.join(map(lambda x: str(x), score_z_disc_tar1_metrics)))
        print('  score_z_disc2: %s' % ', '.join(map(lambda x: str(x), score_z_disc2_metrics)))
        print('  score_z_disc_tar2: %s' % ', '.join(map(lambda x: str(x), score_z_disc_tar2_metrics)))
        print('  score_z_disc3: %s' % ', '.join(map(lambda x: str(x), score_z_disc3_metrics)))
        print('  score_z_disc_tar3: %s' % ', '.join(map(lambda x: str(x), score_z_disc_tar3_metrics)))

    score_z_cont_metrics = []
    score_z_disc1_metrics = []
    score_z_disc2_metrics = []
    score_z_disc3_metrics = []
    score_z_disc_metrics = []

    for k in range(1):
        z_guassian = torch.randn(batch_size, n_latent).cuda()
        z_discrete_1 = torch.from_numpy(np.random.choice(n_discrete_1, batch_size, p=prob_discrete_1)).cuda()
        z_discrete_2 = torch.from_numpy(np.random.choice(n_discrete_2, batch_size, p=prob_discrete_2)).cuda()
        z_discrete_3 = torch.from_numpy(np.random.choice(n_discrete_3, batch_size, p=prob_discrete_3)).cuda()
        x_samples = generator(z_guassian, z_discrete_1, z_discrete_2, z_discrete_3)
        z_cont, z_disc1, z_disc2, z_disc3 = encoder(x_samples.detach())
        z_disc = torch.cat((z_disc1, z_disc2, z_disc3), 1)
        z_disc1_grp = z_disc1.reshape(-1, group_size * z_disc1.size(1))
        z_disc2_grp = z_disc2.reshape(-1, group_size * z_disc2.size(1))
        z_disc3_grp = z_disc3.reshape(-1, group_size * z_disc3.size(1))
        z_disc_grp = z_disc.reshape(-1, group_size // 2 * z_disc.size(1))

        score_z_cont = discr_cont(z_cont).mean()
        score_z_disc1 = discr_disc1(z_disc1_grp).mean()
        score_z_disc2 = discr_disc2(z_disc2_grp).mean()
        score_z_disc3 = discr_disc3(z_disc3_grp).mean()
        score_z_disc = discr_disc(z_disc_grp).mean()
        score_z_cont_metrics.append(score_z_cont.detach().cpu().item())
        score_z_disc1_metrics.append(score_z_disc1.detach().cpu().item())
        score_z_disc2_metrics.append(score_z_disc2.detach().cpu().item())
        score_z_disc3_metrics.append(score_z_disc3.detach().cpu().item())
        score_z_disc_metrics.append(score_z_disc.detach().cpu().item())

        x_pred = decoder(z_cont, z_disc1, z_disc2, z_disc3)
        recon_loss = F.mse_loss(x_pred, x_samples)
        encdec_loss = recon_loss - score_z_cont - score_z_disc1 - score_z_disc2 - score_z_disc3 - score_z_disc

        encdec_optimzier.zero_grad()
        encdec_loss.backward()
        encdec_optimzier.step()

    if step % 100 == 0:
        print('Step %d [Training Encoder & Decoder]' % step)
        print('  score_z_cont: %s' % ', '.join(map(lambda x: str(x), score_z_cont_metrics)))
        print('  score_z_disc: %s' % ', '.join(map(lambda x: str(x), score_z_disc_metrics)))
        print('  score_z_disc1: %s' % ', '.join(map(lambda x: str(x), score_z_disc1_metrics)))
        print('  score_z_disc2: %s' % ', '.join(map(lambda x: str(x), score_z_disc2_metrics)))
        print('  score_z_disc3: %s' % ', '.join(map(lambda x: str(x), score_z_disc3_metrics)))


my_z_cont, my_z_disc1, my_z_disc2, my_z_disc3 = encoder(my_x_samples)
my_x_pred = decoder(my_z_cont, my_z_disc1, my_z_disc2, my_z_disc3)

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
my_z_discrete_1 = F.one_hot(my_z_discrete_1, n_discrete_1).float()
my_z_discrete_2 = F.one_hot(my_z_discrete_2, n_discrete_2).float()
my_z_discrete_3 = F.one_hot(my_z_discrete_3, n_discrete_3).float()
tsne(encoder.z_discrete_p[0], 'my_z_disc_p1_tsne')
tsne(encoder.z_discrete_p[1], 'my_z_disc_p2_tsne')
tsne(encoder.z_discrete_p[2], 'my_z_disc_p3_tsne')
print('my_z_disc1', my_z_disc1.detach().sum(0).cpu())
print('my_z_discrete_1', my_z_discrete_1.detach().sum(0).cpu())
print('my_z_disc2', my_z_disc2.detach().sum(0).cpu())
print('my_z_discrete_2', my_z_discrete_2.detach().sum(0).cpu())
print('my_z_disc3', my_z_disc3.detach().sum(0).cpu())
print('my_z_discrete_3', my_z_discrete_3.detach().sum(0).cpu())

my_z_discrete = torch.cat((my_z_discrete_1, my_z_discrete_2, my_z_discrete_3), 1).t() * 2 - 1
my_z_disc = torch.cat((my_z_disc1, my_z_disc2, my_z_disc3), 1).t() * 2 - 1
my_cov = np.cov(my_z_discrete.detach().cpu().numpy(), my_z_disc.detach().cpu().numpy())
np.savetxt('my_cov.txt', my_cov, fmt='%8.3f', delimiter=',')