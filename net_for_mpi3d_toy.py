import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import utils as U


class Mpi3dToy(object):
    def __init__(self):
        self.data = np.load('mpi3d_toy/mpi3d_toy.npz')['images'].reshape([4, 4, 2, 3, 3, 40, 40, 64, 64, 3])
        self.factors = {'color': 4, 'shape': 4, 'size': 2, 'camera': 3, 'background': 3, 'horizontal': 40, 'vertical': 40}

    def get_samples(self, n_samples):
        sampled_factors = tuple([np.random.randint(0, self.factors[name], n_samples)
                                 for name in ('color', 'shape', 'size', 'camera', 'background', 'horizontal', 'vertical')])
        sampled_images = self.data[sampled_factors]
        sampled_factors = np.stack(sampled_factors, axis=1)
        return sampled_factors, sampled_images

    def show_images(self, nr=5, nc=5):
        n_samples = nr * nc
        factors, images = self.draw_samples(n_samples)
        fig, axs = plt.subplots(nr, nc, figsize=(20, 10))
        for ax, fac, img in zip(axs.ravel(), factors, images):
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(str(fac), fontsize=10)
        plt.show()
        fig.savefig('mpi3dtoy_images.pdf')


mpi3d_toy = Mpi3dToy()
#mpi3d_toy.show_images()
#factor_samples, image_samples = mpi3d_toy.get_samples(10)


class Net(nn.Module):
    ''' 3 x 64 x 64 ===> 16 x 8 x 8
        Path 1: 3 x 64 x 64 => conv3x16x5x5,+4 (16x16x16) => conv16x16x3x3,+2 => 16 x 8 x 8 => 1024
        Path 2: 3 x 64 x 64 => conv3x8x2x2,+2 (8x32x32) => conv8x16x2x2,+2 (16x16x16) => conv16x32x2x2,+2 (32x8x8) => conv32x64x2x2,+2 => 64 x 4 x 4 => 1024
        Path 3: 3 x 64 x 64 => conv3x16x6x6,+4 (16x16x16) => conv16x64x4x4,+2 (64x8x8) => conv64x256x1x1,+1 (256x8x8) => avgpool (256) => fc => 1024
        Path 4: 3 x 64 x 64 => conv3x32x4x4,+2 (32x32x32), maxpool4x4,+2 (32x16x16) => conv32x64x4x4,+2 (64x8x8) => conv64x256x1x1,+1 (256x8x8) => avgpool (256) => fc => 1024
    '''

    def __init__(self, mpi3d_toy):
        super(Net, self).__init__()
        self.mpi3d_toy = mpi3d_toy
        self.path = nn.Sequential(U.conv2d(3, 16, 6, 4, True, 64), nn.ReLU(),  # 16 x 16 x 16
                                  U.conv2d(16, 64, 4, 2, True, 16), nn.ReLU())  # 64 x 8 x 8
        self.path_a = nn.Sequential(U.conv2d(64, 16, 1, 1, True, 8), nn.ReLU(),  # 16 x 8 x 8
                                    U.Lambda(lambda x: x.reshape(-1, 1024)))  # 1024
        self.path_b = nn.Sequential(U.conv2d(64, 64, 4, 2, True, 8), nn.ReLU(),  # 64 x 4 x 4
                                    U.conv2d(64, 64, 4, 1, True, 4), nn.ReLU(),  # 64 x 4 x 4
                                    U.Lambda(lambda x: x.reshape(-1, 1024)))  # 1024
        self.path_c = nn.Sequential(U.conv2d(64, 256, 1, 1, True, 8), nn.ReLU(),  # 256 x 8 x 8
                                    nn.AdaptiveMaxPool2d((1, 1)), U.Lambda(lambda x: x.reshape(-1, 256)),  # 256
                                    nn.Linear(256, 1024), nn.ReLU())  # 1024

        self.fc1 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.output_color = nn.Sequential(nn.Linear(256, self.mpi3d_toy.factors['color']))
        self.output_shape = nn.Sequential(nn.Linear(256, 64), nn.ReLU(),
                                          nn.Linear(64, self.mpi3d_toy.factors['shape']))
        self.output_size = nn.Sequential(nn.Linear(256, 64), nn.ReLU(),
                                         nn.Linear(64, self.mpi3d_toy.factors['size']))

        self.fc2 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.output_horizontal = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
                                               nn.Linear(256, self.mpi3d_toy.factors['horizontal']))
        self.output_vertical = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
                                             nn.Linear(256, self.mpi3d_toy.factors['vertical']))

        self.output_camera = nn.Sequential(nn.Linear(1024, self.mpi3d_toy.factors['camera']))
        self.output_background = nn.Sequential(nn.Linear(1024, self.mpi3d_toy.factors['background']))

    def forward(self, images):
        ''' images: B x C x H x W
        '''
        x = self.path(images)
        out_path = self.path_a(x) + self.path_b(x) + self.path_c(x)

        out_fc1 = self.fc1(out_path)
        out_color = self.output_color(out_fc1)
        out_shape = self.output_shape(out_fc1)
        out_size = self.output_size(out_fc1)

        out_fc2 = self.fc2(out_path)
        out_horizontal = self.output_horizontal(out_fc2)
        out_vertical = self.output_vertical(out_fc2)

        out_camera = self.output_camera(out_path)
        out_background = self.output_background(out_path)

        return {'color': out_color, 'shape': out_shape, 'size': out_size,
                'camera': out_camera, 'background': out_background,
                'horizontal': out_horizontal, 'vertical': out_vertical}


net = Net(mpi3d_toy).cuda()


def loss_fn(outputs, factors):
    loss_color = F.cross_entropy(outputs['color'], factors[:, 0])
    loss_shape = F.cross_entropy(outputs['shape'], factors[:, 1])
    loss_size = F.cross_entropy(outputs['size'], factors[:, 2])
    loss_camera = F.cross_entropy(outputs['camera'], factors[:, 3])
    loss_background = F.cross_entropy(outputs['background'], factors[:, 4])
    loss_horizontal = F.cross_entropy(outputs['horizontal'], factors[:, 5])
    loss_vertical = F.cross_entropy(outputs['vertical'], factors[:, 6])
    return loss_color + loss_shape + loss_size + loss_camera + loss_background + loss_horizontal + loss_vertical


def accuracy_fn(outputs, factors):
    with torch.no_grad():
        acc_color = torch.eq(outputs['color'].argmax(1), factors[:, 0]).float().mean().cpu().item()
        acc_shape = torch.eq(outputs['shape'].argmax(1), factors[:, 1]).float().mean().cpu().item()
        acc_size = torch.eq(outputs['size'].argmax(1), factors[:, 2]).float().mean().cpu().item()
        acc_camera = torch.eq(outputs['camera'].argmax(1), factors[:, 3]).float().mean().cpu().item()
        acc_background = torch.eq(outputs['background'].argmax(1), factors[:, 4]).float().mean().cpu().item()
        acc_horizontal = torch.eq(outputs['horizontal'].argmax(1), factors[:, 5]).float().mean().cpu().item()
        acc_vertical = torch.eq(outputs['vertical'].argmax(1), factors[:, 6]).float().mean().cpu().item()

    return {'color': acc_color, 'shape': acc_shape, 'size': acc_size,
            'camera': acc_camera, 'background': acc_background,
            'horizontal': acc_horizontal, 'vertical': acc_vertical}


def train():
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    batch_size = 128
    max_steps = 10000
    for step in range(max_steps):
        factor_samples, image_samples = mpi3d_toy.get_samples(batch_size)
        image_samples = torch.from_numpy(np.transpose(image_samples.astype(np.float32) / 255, (0, 3, 1, 2))).cuda()  # B x C x H x W
        factor_samples = torch.from_numpy(factor_samples).cuda()  # B x 7

        outputs = net(image_samples)
        loss = loss_fn(outputs, factor_samples)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            loss = loss.detach().cpu().item()
            accs = accuracy_fn(outputs, factor_samples)
            print('[step %d] loss: %.4f, acc_color: %.4f, acc_shape: %.4f, acc_size: %.4f, acc_camera: %.4f, '
                  'acc_bkg: %.4f, acc_hori: %.4f, acc_vert: %.4f'
                  % (step, loss, accs['color'], accs['shape'], accs['size'], accs['camera'],
                     accs['background'], accs['horizontal'], accs['vertical']))


train()

