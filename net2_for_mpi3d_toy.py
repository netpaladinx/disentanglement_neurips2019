from itertools import chain

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
        self.path_base = nn.Sequential(U.conv2d(3, 16, 6, 4, True, 64), nn.ReLU(),  # 16 x 16 x 16
                                       U.conv2d(16, 64, 4, 2, True, 16), nn.ReLU())  # 64 x 8 x 8
        self.path_shallow = nn.Sequential(U.conv2d(64, 16, 1, 1, True, 8), nn.ReLU(),  # 16 x 8 x 8
                                          U.Lambda(lambda x: x.reshape(-1, 1024)),
                                          nn.Linear(1024, 1024), nn.ReLU())  # 1024
        self.path_deep = nn.Sequential(U.conv2d(64, 64, 4, 2, True, 8), nn.ReLU(),  # 64 x 4 x 4
                                       U.conv2d(64, 64, 4, 1, True, 4), nn.ReLU(),  # 64 x 4 x 4
                                       U.Lambda(lambda x: x.reshape(-1, 1024)),
                                       nn.Linear(1024, 1024), nn.ReLU())  # 1024
        self.path_pool = nn.Sequential(U.conv2d(64, 256, 1, 1, True, 8), nn.ReLU(),  # 256 x 8 x 8
                                       nn.AdaptiveMaxPool2d((1, 1)), U.Lambda(lambda x: x.reshape(-1, 256)),  # 256
                                       nn.Linear(256, 1024), nn.ReLU())  # 1024

        self.path_base2 = nn.Sequential(U.conv2d(3, 16, 2, 2, True, 64), nn.ReLU(),  # 16 x 32 x 32
                                        U.conv2d(16, 64, 2, 2, True, 16), nn.ReLU())  # 64 x 16 x 16
        self.path_shallow2 = nn.Sequential(nn.MaxPool2d(2, 2),  # 64 x 8 x 8
                                           U.conv2d(64, 16, 1, 1, True, 8), nn.ReLU(),  # 16 x 8 x 8
                                           U.Lambda(lambda x: x.reshape(-1, 1024)),
                                           nn.Linear(1024, 1024), nn.ReLU())  # 1024
        self.path_deep2 = nn.Sequential(U.conv2d(64, 64, 4, 2, True, 16), nn.ReLU(),  # 64 x 8 x 8
                                        U.conv2d(64, 64, 4, 2, True, 8), nn.ReLU(),  # 64 x 4 x 4
                                        U.Lambda(lambda x: x.reshape(-1, 1024)),
                                        nn.Linear(1024, 1024), nn.ReLU())  # 1024
        self.path_pool2 = nn.Sequential(U.conv2d(64, 256, 2, 2, True, 16), nn.ReLU(),  # 256 x 8 x 8
                                        nn.AdaptiveMaxPool2d((1, 1)), U.Lambda(lambda x: x.reshape(-1, 256)),  # 256
                                        nn.Linear(256, 1024), nn.ReLU())  # 1024

        self.weight_col_shp_siz = nn.Parameter(torch.ones(6))
        self.path_col_shp_siz_shallow = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.path_col_shp_siz_deep = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                                   nn.Linear(512, 256), nn.ReLU())
        self.path_col_shp_siz_deeper = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                                     nn.Linear(512, 256), nn.ReLU(),
                                                     nn.Linear(256, 256), nn.ReLU())

        self.weight_col = nn.Parameter(torch.ones(3))
        self.weight_shp = nn.Parameter(torch.ones(3))
        self.weight_siz = nn.Parameter(torch.ones(3))
        self.output_color = nn.Sequential(nn.Linear(256, self.mpi3d_toy.factors['color']))
        self.output_shape = nn.Sequential(nn.Linear(256, self.mpi3d_toy.factors['shape']))
        self.output_size = nn.Sequential(nn.Linear(256, self.mpi3d_toy.factors['size']))

        self.weight_hor_ver = nn.Parameter(torch.ones(6))
        self.path_hor_ver_shallow = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.path_hor_ver_deep = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                               nn.Linear(512, 256), nn.ReLU())
        self.path_hor_ver_deeper = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                                 nn.Linear(512, 256), nn.ReLU(),
                                                 nn.Linear(256, 256), nn.ReLU())

        self.weight_hor = nn.Parameter(torch.ones(3))
        self.weight_ver = nn.Parameter(torch.ones(3))
        self.output_horizontal = nn.Sequential(nn.Linear(256, self.mpi3d_toy.factors['horizontal']))
        self.output_vertical = nn.Sequential(nn.Linear(256, self.mpi3d_toy.factors['vertical']))

        self.weight_cam = nn.Parameter(torch.ones(6))
        self.weight_bkg = nn.Parameter(torch.ones(6))
        self.output_camera = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(),
                                           nn.Linear(256, self.mpi3d_toy.factors['camera']))
        self.output_background = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(),
                                               nn.Linear(256, self.mpi3d_toy.factors['background']))

    def weight_fn(self, x):
        return F.leaky_relu(x)

    def print_weights(self):
        to_numpy = lambda x: self.weight_fn(x).detach().cpu().numpy()
        print('  weight_col_shp_siz:', to_numpy(self.weight_col_shp_siz))
        print('  weight_col:', to_numpy(self.weight_col))
        print('  weight_shp:', to_numpy(self.weight_shp))
        print('  weight_siz:', to_numpy(self.weight_siz))
        print('  weight_hor_ver:', to_numpy(self.weight_hor_ver))
        print('  weight_hor:', to_numpy(self.weight_hor))
        print('  weight_ver:', to_numpy(self.weight_ver))
        print('  weight_cam:', to_numpy(self.weight_cam))
        print('  weight_bkg:', to_numpy(self.weight_bkg))

    def forward(self, images):
        ''' images: B x C x H x W
        '''
        x = self.path_base(images)
        out_shallow = self.path_shallow(x)
        out_deep = self.path_deep(x)
        out_pool = self.path_pool(x)

        x = self.path_base2(images)
        out_shallow2 = self.path_shallow2(x)
        out_deep2 = self.path_deep2(x)
        out_pool2 = self.path_pool2(x)

        weight = self.weight_fn(self.weight_col_shp_siz)
        out_css = (out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2] +
                   out_shallow2 * weight[3] + out_deep2 * weight[4] + out_pool2 * weight[5])
        out_css_shallow = self.path_col_shp_siz_shallow(out_css)
        out_css_deep = self.path_col_shp_siz_deep(out_css)
        out_css_deeper = self.path_col_shp_siz_deeper(out_css)

        weight = self.weight_fn(self.weight_col)
        out_color = out_css_shallow * weight[0] + out_css_deep * weight[1] + out_css_deeper * weight[2]
        out_color = self.output_color(out_color)

        weight = self.weight_fn(self.weight_shp)
        out_shape = out_css_shallow * weight[0] + out_css_deep * weight[1] + out_css_deeper * weight[2]
        out_shape = self.output_shape(out_shape)

        weight = self.weight_fn(self.weight_siz)
        out_size = out_css_shallow * weight[0] + out_css_deep * weight[1] + out_css_deeper * weight[2]
        out_size = self.output_size(out_size)

        weight = self.weight_fn(self.weight_hor_ver)
        out_hv = (out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2] +
                  out_shallow2 * weight[3] + out_deep2 * weight[4] + out_pool2 * weight[5])
        out_hv_shallow = self.path_hor_ver_shallow(out_hv)
        out_hv_deep = self.path_hor_ver_deep(out_hv)
        out_hv_deeper = self.path_hor_ver_deeper(out_hv)

        weight = self.weight_fn(self.weight_hor)
        out_horizontal = out_hv_shallow * weight[0] + out_hv_deep * weight[1] + out_hv_deeper * weight[2]
        out_horizontal = self.output_horizontal(out_horizontal)

        weight = self.weight_fn(self.weight_ver)
        out_vertical = out_hv_shallow * weight[0] + out_hv_deep * weight[1] + out_hv_deeper * weight[2]
        out_vertical = self.output_vertical(out_vertical)

        weight = self.weight_fn(self.weight_cam)
        out_camera = (out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2] +
                      out_shallow2 * weight[3] + out_deep2 * weight[4] + out_pool2 * weight[5])
        out_camera = self.output_camera(out_camera)

        weight = self.weight_fn(self.weight_bkg)
        out_background = (out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2] +
                          out_shallow2 * weight[3] + out_deep2 * weight[4] + out_pool2 * weight[5])
        out_background = self.output_background(out_background)

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


def accuracy_fn(outputs, factors, accs=None):
    with torch.no_grad():
        acc_color = torch.eq(outputs['color'].argmax(1), factors[:, 0]).float().mean().cpu().item()
        acc_shape = torch.eq(outputs['shape'].argmax(1), factors[:, 1]).float().mean().cpu().item()
        acc_size = torch.eq(outputs['size'].argmax(1), factors[:, 2]).float().mean().cpu().item()
        acc_camera = torch.eq(outputs['camera'].argmax(1), factors[:, 3]).float().mean().cpu().item()
        acc_background = torch.eq(outputs['background'].argmax(1), factors[:, 4]).float().mean().cpu().item()
        acc_horizontal = torch.eq(outputs['horizontal'].argmax(1), factors[:, 5]).float().mean().cpu().item()
        acc_vertical = torch.eq(outputs['vertical'].argmax(1), factors[:, 6]).float().mean().cpu().item()

    return {'color': acc_color if accs is None else acc_color * 0.1 + accs['color'] * 0.9,
            'shape': acc_shape if accs is None else acc_shape * 0.1 + accs['shape'] * 0.9,
            'size': acc_size if accs is None else acc_size * 0.1 + accs['size'] * 0.9,
            'camera': acc_camera if accs is None else acc_camera * 0.1 + accs['camera'] * 0.9,
            'background': acc_background if accs is None else acc_background * 0.1 + accs['background'] * 0.9,
            'horizontal': acc_horizontal if accs is None else acc_horizontal * 0.1 + accs['horizontal'] * 0.9,
            'vertical': acc_vertical if accs is None else acc_vertical * 0.1 + accs['vertical'] * 0.9}


def preprocess(images):
    images = np.transpose(images, [0, 3, 1, 2])  # B x C x H x W
    mu = images.mean()
    sigma = images.std()
    images = (images - mu) / (sigma + 1e-10)
    return torch.from_numpy(images.astype(np.float32))


def code_transformer(codes):
    codes_prob, codes_oh, codes_oh_prob, codes_oh_trans, codes_oh_scale, codes_sg, codes_hg = {}, {}, {}, {}, {}, {}, {}
    for name in codes:
        logits = codes[name]
        prob = torch.softmax(logits, 1)  # B x n_values

        argmax = torch.argmax(logits, 1, True)  # B
        one_hot = torch.zeros_like(logits).scatter_(1, argmax, 1.0)
        one_hot_prob = prob * one_hot
        one_hot_trans = one_hot - one_hot_prob.detach() + one_hot_prob
        one_hot_scale = one_hot_prob / one_hot_prob.sum(1, keepdim=True).detach()

        soft_gumbel, hard_gumbel, _ = U.gumbel_softmax(logits, tau=1, dim=1)

        codes_prob[name] = prob
        codes_oh[name] = one_hot
        codes_oh_prob[name] = one_hot_prob
        codes_oh_trans[name] = one_hot_trans
        codes_oh_scale[name] = one_hot_scale
        codes_sg[name] = soft_gumbel
        codes_hg[name] = hard_gumbel

    return {'prob': codes_prob, 'one_hot': codes_oh, 'one_hot_prob': codes_oh_prob,
            'one_hot_trans': codes_oh_trans, 'one_hot_scale': codes_oh_scale,
            'soft_gumbel': codes_sg, 'hard_gumbel': codes_hg}


class CodeCompressor(nn.Module):
    def __init__(self, n_values):
        super(CodeCompressor, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(n_values, n_values), nn.LeakyReLU(),
                                     nn.Linear(n_values, 1), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(1, n_values), nn.LeakyReLU(),
                                     nn.Linear(n_values, n_values))

    def forward(self, x):
        z = self.encoder(x)
        pred_x = self.decoder(z)
        loss = F.mse_loss(pred_x, x, reduction='none').sum(1).mean()
        return loss, z, pred_x


code_compressors = {name: CodeCompressor(n_values).cuda() for name, n_values in mpi3d_toy.factors.items()}


def train():
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    compressors_optimizer = optim.Adam(chain(*[code_compressors[name].parameters() for name in code_compressors]), lr=0.0001)

    batch_size = 128
    max_steps = 30000
    accs = None
    for step in range(max_steps):
        factor_samples, image_samples = mpi3d_toy.get_samples(batch_size)
        image_samples = preprocess(image_samples).cuda()  # B x C x H x W
        factor_samples = torch.from_numpy(factor_samples).cuda()  # B x 7

        outputs = net(image_samples)
        loss = loss_fn(outputs, factor_samples)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            loss = loss.detach().cpu().item()
            accs = accuracy_fn(outputs, factor_samples, accs)
            print('[step %d] loss: %.4f, acc_color: %.4f, acc_shape: %.4f, acc_size: %.4f, acc_camera: %.4f, '
                  'acc_bkg: %.4f, acc_hori: %.4f, acc_vert: %.4f'
                  % (step, loss, accs['color'], accs['shape'], accs['size'], accs['camera'],
                     accs['background'], accs['horizontal'], accs['vertical']))
            net.print_weights()

        z_factors = {}
        codes = code_transformer(outputs)
        loss = 0
        for name, oh_code in codes['one_hot'].items():
            loss_oh, z, _ = code_compressors[name](oh_code.detach())
            z_factors[name] = z
            loss = loss + loss_oh
        compressors_optimizer.zero_grad()
        loss.backward()
        compressors_optimizer.step()

        if step % 100 == 0:
            loss = loss.detach().cpu().item()
            print('[step %d] loss: %.4f' % (step, loss))


train()


from disentanglement_lib.data.ground_truth import mpi3d
from disentanglement_lib.evaluation.metrics import dci, factor_vae, sap_score, mig, irs, utils


def hard_representation_fn(x):
    with torch.no_grad():
        images = preprocess(x).cuda()
        codes_logits = net(images)
        transformed_codes = code_transformer(codes_logits)

        z_factors = {}
        for name, oh_code in transformed_codes['one_hot'].items():
            _, z, _ = code_compressors[name](oh_code)
            z_factors[name] = z

        repr_code = torch.cat([z for k, z in z_factors.items()], 1).cpu().numpy()
    return repr_code

def soft_representation_fn(x):
    with torch.no_grad():
        images = preprocess(x).cuda()
        codes_logits = net(images)
        transformed_codes = code_transformer(codes_logits)

        z_factors = {}
        for name, oh_code in transformed_codes['prob'].items():
            _, z, _ = code_compressors[name](oh_code)
            z_factors[name] = z

        repr_code = torch.cat([z for k, z in z_factors.items()], 1).cpu().numpy()
    return repr_code

ground_truth_data = mpi3d.MPI3D()

print('hard_representation_fn')
scores = factor_vae.compute_factor_vae(ground_truth_data, hard_representation_fn, np.random.RandomState(0), 64, 10000, 5000, 10000)
print('  factor_vae: %.6f' % scores['eval_accuracy'])

scores = dci.compute_dci(ground_truth_data, hard_representation_fn, np.random.RandomState(0), 10000, 5000)
print('  dci: %.6f' % scores['disentanglement'])

scores = sap_score.compute_sap(ground_truth_data, hard_representation_fn, np.random.RandomState(0), 10000, 5000, continuous_factors=False)
print('  sap_score: %.6f' % scores['SAP_score'])

import gin.tf
gin.bind_parameter("discretizer.discretizer_fn", utils._histogram_discretize)
gin.bind_parameter("discretizer.num_bins", 20)

scores = mig.compute_mig(ground_truth_data, hard_representation_fn, np.random.RandomState(0), 10000)
print('  mig: %.6f' % scores['discrete_mig'])

gin.bind_parameter("irs.batch_size", 16)
scores = irs.compute_irs(ground_truth_data, hard_representation_fn, np.random.RandomState(0), num_train=10000)
print('  irs: %.6f' % scores['IRS'])

print('soft_representation_fn')
scores = factor_vae.compute_factor_vae(ground_truth_data, soft_representation_fn, np.random.RandomState(0), 64, 10000, 5000, 10000)
print('  factor_vae: %.6f' % scores['eval_accuracy'])

scores = dci.compute_dci(ground_truth_data, soft_representation_fn, np.random.RandomState(0), 10000, 5000)
print('  dci: %.6f' % scores['disentanglement'])

scores = sap_score.compute_sap(ground_truth_data, soft_representation_fn, np.random.RandomState(0), 10000, 5000, continuous_factors=False)
print('  sap_score: %.6f' % scores['SAP_score'])

import gin.tf
gin.bind_parameter("discretizer.discretizer_fn", utils._histogram_discretize)
gin.bind_parameter("discretizer.num_bins", 20)

scores = mig.compute_mig(ground_truth_data, soft_representation_fn, np.random.RandomState(0), 10000)
print('  mig: %.6f' % scores['discrete_mig'])

gin.bind_parameter("irs.batch_size", 16)
scores = irs.compute_irs(ground_truth_data, soft_representation_fn, np.random.RandomState(0), num_train=10000)
print('  irs: %.6f' % scores['IRS'])