import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils as U


N_EMBED_DIMS = 32
N_FACTOR_DIMS = 256

class Encoder(nn.Module):
    def __init__(self, factors):
        super(Encoder, self).__init__()
        self.factors = factors
        self.path_base = nn.Sequential(U.conv2d(3, 16, 6, 4, True, 64), nn.ReLU(),  # 16 x 16 x 16
                                       U.conv2d(16, 64, 4, 2, True, 16), nn.ReLU())  # 64 x 8 x 8
        self.path_shallow = nn.Sequential(U.conv2d(64, 16, 1, 1, True, 8), nn.ReLU(),  # 16 x 8 x 8
                                          U.Lambda(lambda x: x.reshape(-1, 1024)),  # 1024
                                          nn.Linear(1024, 1024), nn.ReLU())
        self.path_deep = nn.Sequential(U.conv2d(64, 64, 4, 2, True, 8), nn.ReLU(),  # 64 x 4 x 4
                                       U.conv2d(64, 64, 4, 1, True, 4), nn.ReLU(),  # 64 x 4 x 4
                                       U.Lambda(lambda x: x.reshape(-1, 1024)),  # 1024
                                       nn.Linear(1024, 1024), nn.ReLU())
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

        self.weight_col_shp_siz = [0.5, 0.0, 1.5, 1.0, 0.0, 2.0]
        self.path_col_shp_siz_shallow = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.path_col_shp_siz_deep = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                                   nn.Linear(512, 256), nn.ReLU())
        self.path_col_shp_siz_deeper = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                                     nn.Linear(512, 256), nn.ReLU(),
                                                     nn.Linear(256, 256), nn.ReLU())

        self.weight_col = [1.0, 0.0, 0.5]
        self.weight_shp = [0.0, 0.5, 1.0]
        self.weight_siz = [0.0, 1.0, 0.5]
        self.output_color = nn.ModuleDict({'code_logits': nn.Linear(256, self.factors['color']),
                                           'embed_mu': nn.Linear(256, N_EMBED_DIMS),
                                           'embed_logvar': nn.Linear(256, N_EMBED_DIMS)})
        self.output_shape = nn.ModuleDict({'code_logits': nn.Linear(256, self.factors['shape']),
                                           'embed_mu': nn.Linear(256, N_EMBED_DIMS),
                                           'embed_logvar': nn.Linear(256, N_EMBED_DIMS)})
        self.output_size = nn.ModuleDict({'code_logits': nn.Linear(256, self.factors['size']),
                                          'embed_mu': nn.Linear(256, N_EMBED_DIMS),
                                          'embed_logvar': nn.Linear(256, N_EMBED_DIMS)})

        self.weight_hor_ver = [1.0, 0.0, 0.0, 2.0, 1.5, 0.5]
        self.path_hor_ver_shallow = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.path_hor_ver_deep = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                               nn.Linear(512, 256), nn.ReLU())
        self.path_hor_ver_deeper = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                                 nn.Linear(512, 256), nn.ReLU(),
                                                 nn.Linear(256, 256), nn.ReLU())

        self.weight_hor = [0.5, 0.0, 1.0]
        self.weight_ver = [0.5, 1.0, 0.0]
        self.output_horizontal = nn.ModuleDict({'code_logits': nn.Linear(256, self.factors['horizontal']),
                                                'embed_mu': nn.Linear(256, N_EMBED_DIMS),
                                                'embed_logvar': nn.Linear(256, N_EMBED_DIMS)})
        self.output_vertical = nn.ModuleDict({'code_logits': nn.Linear(256, self.factors['vertical']),
                                              'embed_mu': nn.Linear(256, N_EMBED_DIMS),
                                              'embed_logvar': nn.Linear(256, N_EMBED_DIMS)})

        self.weight_cam = [0.0, 0.5, 0.0, 1.0, 1.5, 0.0]
        self.weight_bkg = [0.5, 1.5, 0.0, 0.0, 0.0, 1.0]
        self.output_camera = nn.ModuleDict({'code_logits': nn.Linear(1024, self.factors['camera']),
                                            'embed_mu': nn.Linear(1024, N_EMBED_DIMS),
                                            'embed_logvar': nn.Linear(1024, N_EMBED_DIMS)})
        self.output_background = nn.ModuleDict({'code_logits': nn.Linear(1024, self.factors['background']),
                                                'embed_mu': nn.Linear(1024, N_EMBED_DIMS),
                                                'embed_logvar': nn.Linear(1024, N_EMBED_DIMS)})

    def regularizer(self, embeds_mu, embeds_logvar):
        reg_loss = 0
        for name in embeds_mu:
            mu = embeds_mu[name]
            logvar = embeds_logvar[name]
            reg_loss += - 0.5 * (1 + logvar - logvar.exp() - mu.pow(2)).sum(1).mean()
        return reg_loss

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

        weight = self.weight_col_shp_siz
        in_css = (out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2] +
                  out_shallow2 * weight[3] + out_deep2 * weight[4] + out_pool2 * weight[5])
        out_css_shallow = self.path_col_shp_siz_shallow(in_css)
        out_css_deep = self.path_col_shp_siz_deep(in_css)
        out_css_deeper = self.path_col_shp_siz_deeper(in_css)

        weight = self.weight_col
        in_color = out_css_shallow * weight[0] + out_css_deep * weight[1] + out_css_deeper * weight[2]
        color_code_logits = self.output_color['code_logits'](in_color)
        color_embed_mu = self.output_color['embed_mu'](in_color)
        color_embed_logvar = self.output_color['embed_logvar'](in_color)

        weight = self.weight_shp
        in_shape = out_css_shallow * weight[0] + out_css_deep * weight[1] + out_css_deeper * weight[2]
        shape_code_logits = self.output_shape['code_logits'](in_shape)
        shape_embed_mu = self.output_shape['embed_mu'](in_shape)
        shape_embed_logvar = self.output_shape['embed_logvar'](in_shape)

        weight = self.weight_siz
        in_size = out_css_shallow * weight[0] + out_css_deep * weight[1] + out_css_deeper * weight[2]
        size_code_logits = self.output_size['code_logits'](in_size)
        size_embed_mu = self.output_size['embed_mu'](in_size)
        size_embed_logvar = self.output_size['embed_logvar'](in_size)

        weight = self.weight_hor_ver
        in_hv = (out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2] +
                 out_shallow2 * weight[3] + out_deep2 * weight[4] + out_pool2 * weight[5])
        out_hv_shallow = self.path_hor_ver_shallow(in_hv)
        out_hv_deep = self.path_hor_ver_deep(in_hv)
        out_hv_deeper = self.path_hor_ver_deeper(in_hv)

        weight = self.weight_hor
        in_horizontal = out_hv_shallow * weight[0] + out_hv_deep * weight[1] + out_hv_deeper * weight[2]
        horizontal_code_logits = self.output_horizontal['code_logits'](in_horizontal)
        horizontal_embed_mu = self.output_horizontal['embed_mu'](in_horizontal)
        horizontal_embed_logvar = self.output_horizontal['embed_logvar'](in_horizontal)

        weight = self.weight_ver
        in_vertical = out_hv_shallow * weight[0] + out_hv_deep * weight[1] + out_hv_deeper * weight[2]
        vertical_code_logits = self.output_vertical['code_logits'](in_vertical)
        vertical_embed_mu = self.output_vertical['embed_mu'](in_vertical)
        vertical_embed_logvar = self.output_vertical['embed_logvar'](in_vertical)

        weight = self.weight_cam
        in_camera = (out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2] +
                     out_shallow2 * weight[3] + out_deep2 * weight[4] + out_pool2 * weight[5])
        camera_code_logits = self.output_camera['code_logits'](in_camera)
        camera_embed_mu = self.output_camera['embed_mu'](in_camera)
        camera_embed_logvar = self.output_camera['embed_logvar'](in_camera)

        weight = self.weight_bkg
        in_background = (out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2] +
                         out_shallow2 * weight[3] + out_deep2 * weight[4] + out_pool2 * weight[5])
        background_code_logits = self.output_background['code_logits'](in_background)
        background_embed_mu = self.output_background['embed_mu'](in_background)
        background_embed_logvar = self.output_background['embed_logvar'](in_background)

        codes_logits = {'color': color_code_logits, 'shape': shape_code_logits, 'size': size_code_logits,
                        'camera': camera_code_logits, 'background': background_code_logits,
                        'horizontal': horizontal_code_logits, 'vertical': vertical_code_logits}
        embeds_mu = {'color': color_embed_mu, 'shape': shape_embed_mu, 'size': size_embed_mu,
                     'camera': camera_embed_mu, 'background': background_embed_mu,
                     'horizontal': horizontal_embed_mu, 'vertical': vertical_embed_mu}
        embeds_logvar = {'color': color_embed_logvar, 'shape': shape_embed_logvar, 'size': size_embed_logvar,
                     'camera': camera_embed_logvar, 'background': background_embed_logvar,
                     'horizontal': horizontal_embed_logvar, 'vertical': vertical_embed_logvar}
        return codes_logits, embeds_mu, embeds_logvar


class Decoder(nn.Module):
    def __init__(self, factors):
        super(Decoder, self).__init__()
        self.factors = factors

        self.factor_embeds = nn.ParameterDict({
            'color': nn.Parameter(torch.randn(self.factors['color'], N_FACTOR_DIMS)),
            'shape': nn.Parameter(torch.randn(self.factors['shape'], N_FACTOR_DIMS)),
            'size': nn.Parameter(torch.randn(self.factors['size'], N_FACTOR_DIMS)),
            'camera': nn.Parameter(torch.randn(self.factors['camera'], N_FACTOR_DIMS)),
            'background': nn.Parameter(torch.randn(self.factors['background'], N_FACTOR_DIMS)),
            'horizontal': nn.Parameter(torch.randn(self.factors['horizontal'], N_FACTOR_DIMS)),
            'vertical': nn.Parameter(torch.randn(self.factors['vertical'], N_FACTOR_DIMS))
        })

        n_dims = N_EMBED_DIMS + N_FACTOR_DIMS
        self.input_color = nn.Linear(n_dims, 512)
        self.input_shape = nn.Linear(n_dims, 512)
        self.input_size = nn.Linear(n_dims, 512)
        self.path_col_shp_siz = nn.Sequential(nn.ReLU(), nn.Linear(512, 1024))

        self.input_horizontal = nn.Linear(n_dims, 512)
        self.input_vertical = nn.Linear(n_dims, 512)
        self.path_hor_ver = nn.Sequential(nn.ReLU(), nn.Linear(512, 1024))

        self.input_camera = nn.Sequential(nn.Linear(n_dims, 512), nn.ReLU(), nn.Linear(512, 1024))
        self.input_background = nn.Sequential(nn.Linear(n_dims, 512), nn.ReLU(), nn.Linear(512, 1024))

        self.path_shallow = nn.Sequential(nn.ReLU(), nn.Linear(1024, 1024),
                                          U.Lambda(lambda x: x.reshape(-1, 16, 8, 8)),  # 16 x 8 x 8
                                          nn.ReLU(), U.deconv2d(16, 64, 1, 1, True, 8))  # 64 x 8 x 8
        self.path_deep = nn.Sequential(nn.ReLU(), nn.Linear(1024, 1024),
                                       U.Lambda(lambda x: x.reshape(-1, 64, 4, 4)),  # 64 x 4 x 4
                                       nn.ReLU(), U.deconv2d(64, 64, 4, 1, True, 4),  # 64 x 4 x 4
                                       nn.ReLU(), U.deconv2d(64, 64, 4, 2, True, 8))  # 64 x 8 x 8
        self.path_base = nn.Sequential(nn.ReLU(), U.deconv2d(64, 16, 4, 2, True, 16),  # 16 x 16 x 16
                                       nn.ReLU(), U.deconv2d(16, 3, 6, 4, True, 64))  # 3 x 64 x 64

        self.path_shallow2 = nn.Sequential(nn.ReLU(), nn.Linear(1024, 1024),
                                           U.Lambda(lambda x: x.reshape(-1, 16, 8, 8)),  # 16 x 8 x 8
                                           nn.ReLU(), U.deconv2d(16, 64, 2, 2, True, 16))  # 64 x 16 x 16
        self.path_deep2 = nn.Sequential(nn.ReLU(), nn.Linear(1024, 1024),
                                        U.Lambda(lambda x: x.reshape(-1, 64, 4, 4)),  # 64 x 4 x 4
                                        nn.ReLU(), U.deconv2d(64, 64, 4, 2, True, 8),  # 64 x 8 x 8
                                        nn.ReLU(), U.deconv2d(64, 64, 4, 2, True, 16))  # 64 x 16 x 16
        self.path_base2 = nn.Sequential(nn.ReLU(), U.deconv2d(64, 16, 2, 2, True, 32),  # 16 x 32 x 32
                                        nn.ReLU(), U.deconv2d(16, 3, 2, 2, True, 64))  # 3 x 64 x 64

    def sampler(self, mu, logvar):
        if self.training:
            return torch.randn_like(mu) * (logvar / 2).exp() + mu
        else:
            return mu

    def forward(self, codes, embeds_mu, embeds_logvar):
        color_code = codes['color']
        color_embed_mu = embeds_mu['color']
        color_embed_logvar = embeds_logvar['color']
        shape_code = codes['shape']
        shape_embed_mu = embeds_mu['shape']
        shape_embed_logvar = embeds_logvar['shape']
        size_code = codes['size']
        size_embed_mu = embeds_mu['size']
        size_embed_logvar = embeds_logvar['size']
        camera_code = codes['camera']
        camera_embed_mu = embeds_mu['camera']
        camera_embed_logvar = embeds_logvar['camera']
        background_code = codes['background']
        background_embed_mu = embeds_mu['background']
        background_embed_logvar = embeds_logvar['background']
        horizontal_code = codes['horizontal']
        horizontal_embed_mu = embeds_mu['horizontal']
        horizontal_embed_logvar = embeds_logvar['horizontal']
        vertical_code = codes['vertical']
        vertical_embed_mu = embeds_mu['vertical']
        vertical_embed_logvar = embeds_logvar['vertical']

        color_factor_embed = color_code.mm(self.factor_embeds['color'])
        color_embed = torch.cat((color_factor_embed,
                                 self.sampler(color_embed_mu, color_embed_logvar)), 1)
        color_embed = self.input_color(color_embed)

        shape_factor_embed = shape_code.mm(self.factor_embeds['shape'])
        shape_embed = torch.cat((shape_factor_embed,
                                 self.sampler(shape_embed_mu, shape_embed_logvar)), 1)
        shape_embed = self.input_shape(shape_embed)

        size_factor_embed = size_code.mm(self.factor_embeds['size'])
        size_embed = torch.cat((size_factor_embed,
                                self.sampler(size_embed_mu, size_embed_logvar)), 1)
        size_embed = self.input_size(size_embed)

        col_shp_siz_embed = color_embed + shape_embed + size_embed
        col_shp_siz_embed = self.path_col_shp_siz(col_shp_siz_embed)

        horizontal_factor_embed = horizontal_code.mm(self.factor_embeds['horizontal'])
        horizontal_embed = torch.cat((horizontal_factor_embed,
                                      self.sampler(horizontal_embed_mu, horizontal_embed_logvar)), 1)
        horizontal_embed = self.input_horizontal(horizontal_embed)

        vertical_factor_embed = vertical_code.mm(self.factor_embeds['vertical'])
        vertical_embed = torch.cat((vertical_factor_embed,
                                    self.sampler(vertical_embed_mu, vertical_embed_logvar)), 1)
        vertical_embed = self.input_vertical(vertical_embed)

        hor_ver_embed = horizontal_embed + vertical_embed
        hor_ver_embed = self.path_hor_ver(hor_ver_embed)

        camera_factor_embed = camera_code.mm(self.factor_embeds['camera'])
        camera_embed = torch.cat((camera_factor_embed,
                                  self.sampler(camera_embed_mu, camera_embed_logvar)), 1)
        camera_embed = self.input_camera(camera_embed)

        background_factor_embed = background_code.mm(self.factor_embeds['background'])
        background_embed = torch.cat((background_factor_embed,
                                      self.sampler(background_embed_mu, background_embed_logvar)), 1)
        background_embed = self.input_background(background_embed)

        all_embed = col_shp_siz_embed + hor_ver_embed + camera_embed * 0.5 + background_embed * 0.5
        out_shallow = self.path_shallow(all_embed)
        out_deep = self.path_deep(all_embed)
        out = self.path_base(out_shallow + out_deep)
        out_shallow2 = self.path_shallow2(all_embed)
        out_deep2 = self.path_deep2(all_embed)
        out2 = self.path_base2(out_shallow2 + out_deep2)
        out = out + out2  # B x 3 x 64 x 64
        return out


def preprocess(images):
    images = np.transpose(images, [0, 3, 1, 2])  # B x C x H x W
    mu = images.mean()
    sigma = images.std()
    images = (images - mu) / (sigma + 1e-10)
    return torch.from_numpy(images.astype(np.float32))


def code_transformer(codes, step, max_steps):
    codes_prob, codes_logprob, codes_oh, codes_oh_prob, codes_oh_trans, codes_oh_scale, \
    codes_sg, codes_hg_trans, codes_hg_scale = {}, {}, {}, {}, {}, {}, {}, {}, {}
    alpha = np.power(1 - step / max_steps, 2)
    for name in codes:
        logits = codes[name]
        prob = torch.softmax(logits, 1)  # B x n_values
        logprob = torch.log_softmax(logits, 1)

        argmax = torch.argmax(logits, 1, True)  # B
        one_hot = torch.zeros_like(logits).scatter_(1, argmax, 1.0)
        one_hot_prob = prob * one_hot
        one_hot_trans = one_hot - one_hot_prob.detach() + one_hot_prob
        one_hot_scale = one_hot_prob / one_hot_prob.sum(1, keepdim=True).detach()

        soft_gumbel, hard_gumbel_trans, hard_gumbel_scale = U.gumbel_softmax(logits, tau=1, dim=1)

        codes_prob[name] = prob
        codes_logprob[name] = logprob
        codes_oh[name] = one_hot
        codes_oh_prob[name] = one_hot_prob
        codes_oh_trans[name] = one_hot_trans
        codes_oh_scale[name] = one_hot_scale
        codes_sg[name] = soft_gumbel
        codes_hg_trans[name] = hard_gumbel_trans
        codes_hg_scale[name] = hard_gumbel_scale

    return {'logits': codes, 'prob': codes_prob, 'logprob': codes_logprob,
            'one_hot': codes_oh, 'one_hot_prob': codes_oh_prob,
            'one_hot_trans': codes_oh_trans, 'one_hot_scale': codes_oh_scale,
            'soft_gumbel': codes_sg, 'hard_gumbel_trans': codes_hg_trans, 'hard_gumbel_scale': codes_hg_scale}


class CodeCompressor(nn.Module):
    def __init__(self, n_values):
        super(CodeCompressor, self).__init__()
        n_dims = 2 * n_values
        self.encoder = nn.Sequential(nn.Linear(n_dims, n_dims), nn.LeakyReLU(),
                                     nn.Linear(n_dims, 1), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(1, n_dims), nn.LeakyReLU(),
                                     nn.Linear(n_dims, n_dims))

    def forward(self, x_binary, x_soft):
        x = torch.cat((x_binary, x_soft), 1)
        z = self.encoder(x)
        pred_x = self.decoder(z)
        loss = F.mse_loss(pred_x, x, reduction='none').sum(1).mean()
        return loss, z, pred_x


class Discriminator(nn.Module):
    def __init__(self, n_values, n_groups=8):
        super(Discriminator, self).__init__()
        self.n_values = n_values
        self.n_groups = n_groups
        n_dims = n_values * n_groups
        self.backbone = nn.Sequential(
            nn.Linear(n_values * n_groups, n_dims), nn.LeakyReLU(),
            nn.Linear(n_dims, n_dims), nn.LeakyReLU(),
            nn.Linear(n_dims, n_dims), nn.LeakyReLU(),
            nn.Linear(n_dims, n_dims), nn.LeakyReLU(),
            nn.Linear(n_dims, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)

    def train_loss(self, x):
        batch_size = x.size(0)
        x_tar = F.one_hot(torch.from_numpy(np.random.randint(self.n_values, size=batch_size))).float().cuda()
        x = x.detach().reshape(batch_size // self.n_groups, -1)
        x_tar = x_tar.reshape(batch_size // self.n_groups, -1)
        score_diff = torch.mean(self(x) - self(x_tar))

        t = torch.rand(batch_size // self.n_groups, 1).cuda()
        x_inter = x * t + x_tar * (1 - t)
        x_inter.requires_grad_()
        score_x_inter = self(x_inter)
        score_x_inter.backward(gradient=torch.ones_like(score_x_inter), create_graph=True)
        gradient_penalty = (torch.norm(x_inter.grad, 'fro', 1) - 1).pow(2).mean()

        return score_diff + 10 * gradient_penalty

    def get_score(self, x):
        batch_size = x.size(0)
        x_tar = F.one_hot(torch.from_numpy(np.random.randint(self.n_values, size=batch_size))).float().cuda()
        x = x.detach().reshape(batch_size // self.n_groups, -1)
        x_tar = x_tar.reshape(batch_size // self.n_groups, -1)
        score_x = self(x).mean()
        score_tar = self(x_tar).mean()
        return score_x, score_tar - score_x


def neg_entropy_regularizer(code):
    ''' code: B x n_values
    '''
    prob = code.sum(0) / code.sum()
    entroy = - prob.mul(torch.log(prob + 1e-6)).sum()
    return - entroy


def per_sample_entropy_regularizer(code_prob, code_logprob):
    ''' code: B x n_values
    '''
    entroy = - code_prob.mul(code_logprob).sum(1).mean()
    return entroy


def total_correction_regularizer(code_a, code_b):
    ''' code_a, code_b: B x n_values
    '''
    joint_matrix = code_a.t().mm(code_b)
    joint_matrix = joint_matrix.div(joint_matrix.sum())
    marginal_a = joint_matrix.sum(1, True)
    marginal_b = joint_matrix.sum(0, True)
    return joint_matrix.mul(torch.log(joint_matrix.div(marginal_a * marginal_b + 1e-6) + 1e-6)).sum()

