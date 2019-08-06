import torch
import torch.nn as nn
import torch.nn.functional as F

import utils as U


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.factors = {'color': 4, 'shape': 4, 'size': 2, 'camera': 3, 'background': 3, 'horizontal': 40, 'vertical': 40}
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

        self.weight_col_shp_siz = [0.95, 0.23, 1.33]
        self.path_col_shp_siz_shallow = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.path_col_shp_siz_deep = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                                   nn.Linear(512, 256), nn.ReLU())
        self.path_col_shp_siz_deeper = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                                     nn.Linear(512, 256), nn.ReLU(),
                                                     nn.Linear(256, 256), nn.ReLU())

        self.weight_col = [1.12, 0.53, 0.29]
        self.weight_shp = [0.34, 0.26, 0.29]
        self.weight_siz = [0.36, 0.36, 0.49]
        self.output_color = nn.ModuleDict({'code_logits': nn.Linear(256, self.factors['color']),
                                           'embed_mu': nn.Linear(256, 32),
                                           'embed_logvar': nn.Linear(256, 32)})
        self.output_shape = nn.ModuleDict({'code_logits': nn.Linear(256, self.factors['shape']),
                                           'embed_mu': nn.Linear(256, 32),
                                           'embed_logvar': nn.Linear(256, 32)})
        self.output_size = nn.ModuleDict({'code_logits': nn.Linear(256, self.factors['size']),
                                          'embed_mu': nn.Linear(256, 32),
                                          'embed_logvar': nn.Linear(256, 32)})

        self.weight_hor_ver = [1.36, 0.78, 0.67]
        self.path_hor_ver_shallow = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.path_hor_ver_deep = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                               nn.Linear(512, 256), nn.ReLU())
        self.path_hor_ver_deeper = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                                 nn.Linear(512, 256), nn.ReLU(),
                                                 nn.Linear(256, 256), nn.ReLU())

        self.weight_hor = [0.24, 0.0, 0.14]
        self.weight_ver = [0.30, 0.58, 0.0]
        self.output_horizontal = nn.ModuleDict({'code_logits': nn.Linear(256, self.factors['horizontal']),
                                                'embed_mu': nn.Linear(256, 32),
                                                'embed_logvar': nn.Linear(256, 32)})
        self.output_vertical = nn.ModuleDict({'code_logits': nn.Linear(256, self.factors['vertical']),
                                              'embed_mu': nn.Linear(256, 32),
                                              'embed_logvar': nn.Linear(256, 32)})

        self.weight_cam = [1.35, 1.15, 0.56]
        self.weight_bkg = [1.11, 0.0, 1.54]
        self.output_camera = nn.ModuleDict({'code_logits': nn.Linear(1024, self.factors['camera']),
                                            'embed_mu': nn.Linear(1024, 32),
                                            'embed_logvar': nn.Linear(1024, 32)})
        self.output_background = nn.ModuleDict({'code_logits': nn.Linear(1024, self.factors['background']),
                                                'embed_mu': nn.Linear(1024, 32),
                                                'embed_logvar': nn.Linear(1024, 32)})

    def regularizer(self, embeds_mu, embeds_logvar):
        reg_loss = 0
        for name in embeds_mu:
            mu = embeds_mu[name]  # B x 32
            logvar = embeds_logvar[name]  # B x 32
            reg_loss += - 0.5 * (1 + logvar - logvar.exp() - mu.pow(2)).sum(1).mean()
        return reg_loss

    def forward(self, images):
        ''' images: B x C x H x W
        '''
        x = self.path_base(images)
        out_shallow = self.path_shallow(x)
        out_deep = self.path_deep(x)
        out_pool = self.path_pool(x)

        weight = self.weight_col_shp_siz
        in_css = out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2]
        out_css_shallow = self.path_col_shp_siz_shallow(in_css)
        out_css_deep = self.path_col_shp_siz_deep(in_css)
        out_css_deeper = self.path_col_shp_siz_deeper(in_css)

        weight = self.weight_col
        in_color = out_css_shallow * weight[0] + out_css_deep * weight[1] + out_css_deeper * weight[2]
        color_code_logits = self.output_color['code_logits'](in_color)  # B x 4
        color_embed_mu = self.output_color['embed_mu'](in_color)  # B x 64
        color_embed_logvar = self.output_color['embed_logvar'](in_color)  # B x 64

        weight = self.weight_shp
        in_shape = out_css_shallow * weight[0] + out_css_deep * weight[1] + out_css_deeper * weight[2]
        shape_code_logits = self.output_shape['code_logits'](in_shape)  # B x 4
        shape_embed_mu = self.output_shape['embed_mu'](in_shape)  # B x 64
        shape_embed_logvar = self.output_shape['embed_logvar'](in_shape)  # B x 64

        weight = self.weight_siz
        in_size = out_css_shallow * weight[0] + out_css_deep * weight[1] + out_css_deeper * weight[2]
        size_code_logits = self.output_size['code_logits'](in_size)  # B x 4
        size_embed_mu = self.output_size['embed_mu'](in_size)  # B x 64
        size_embed_logvar = self.output_size['embed_logvar'](in_size)  # B x 64

        weight = self.weight_hor_ver
        in_hv = out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2]
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
        in_camera = out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2]
        camera_code_logits = self.output_camera['code_logits'](in_camera)
        camera_embed_mu = self.output_camera['embed_mu'](in_camera)
        camera_embed_logvar = self.output_camera['embed_logvar'](in_camera)

        weight = self.weight_bkg
        in_background = out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2]
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
    def __init__(self):
        super(Decoder, self).__init__()
        self.factors = {'color': 4, 'shape': 4, 'size': 2, 'camera': 3, 'background': 3, 'horizontal': 40, 'vertical': 40}

        self.mu = nn.ParameterDict({
            'color': nn.Parameter(torch.randn(self.factors['color'], 128)),
            'shape': nn.Parameter(torch.randn(self.factors['shape'], 128)),
            'size': nn.Parameter(torch.randn(self.factors['size'], 128)),
            'camera': nn.Parameter(torch.randn(self.factors['camera'], 128)),
            'background': nn.Parameter(torch.randn(self.factors['background'], 128)),
            'horizontal': nn.Parameter(torch.randn(self.factors['horizontal'], 128)),
            'vertical': nn.Parameter(torch.randn(self.factors['vertical'], 128))
        })
        self.logvar = nn.ParameterDict({
            'color': nn.Parameter(torch.zeros(self.factors['color'], 128)),
            'shape': nn.Parameter(torch.zeros(self.factors['shape'], 128)),
            'size': nn.Parameter(torch.zeros(self.factors['size'], 128)),
            'camera': nn.Parameter(torch.zeros(self.factors['camera'], 128)),
            'background': nn.Parameter(torch.zeros(self.factors['background'], 128)),
            'horizontal': nn.Parameter(torch.zeros(self.factors['horizontal'], 128)),
            'vertical': nn.Parameter(torch.zeros(self.factors['vertical'], 128))
        })

        self.input_color = nn.Linear(160, 256)
        self.input_shape = nn.Linear(160, 256)
        self.input_size = nn.Linear(160, 256)
        self.path_col_shp_siz = nn.Sequential(nn.ReLU(), nn.Linear(256, 1024))

        self.input_horizontal = nn.Linear(160, 256)
        self.input_vertical = nn.Linear(160, 256)
        self.path_hor_ver = nn.Sequential(nn.ReLU(), nn.Linear(256, 1024))

        self.input_camera = nn.Linear(160, 1024)
        self.input_background = nn.Linear(160, 1024)

        self.path_shallow = nn.Sequential(nn.ReLU(), nn.Linear(1024, 1024),
                                          U.Lambda(lambda x: x.reshape(-1, 16, 8, 8)),  # 16 x 8 x 8
                                          nn.ReLU(), U.deconv2d(16, 64, 1, 1, True, 8))  # 64 x 8 x 8
        self.path_deep = nn.Sequential(nn.ReLU(), nn.Linear(1024, 1024),
                                       U.Lambda(lambda x: x.reshape(-1, 64, 4, 4)),  # 64 x 4 x 4
                                       nn.ReLU(), U.deconv2d(64, 64, 4, 1, True, 4),  # 64 x 4 x 4
                                       nn.ReLU(), U.deconv2d(64, 64, 4, 2, True, 8))  # 64 x 8 x 8
        self.path_base = nn.Sequential(nn.ReLU(), U.deconv2d(64, 16, 4, 2, True, 16),  # 16 x 16 x 16
                                       nn.ReLU(), U.deconv2d(16, 3, 6, 4, True, 64))  # 3 x 64 x 64

    def regularizer(self):
        reg_loss = 0
        for name in self.mu:
            mu = self.mu[name]  # B x 128
            logvar = self.logvar[name]  # B x 128
            reg_loss += - 0.5 * (1 + logvar - logvar.exp() - mu.pow(2)).sum(1).mean()
        return reg_loss

    def sampler(self, mu, logvar):
        if self.training:
            return torch.randn_like(mu) * (logvar / 2).exp() + mu
        else:
            return mu

    def forward(self, codes, embeds_mu, embeds_logvar):
        color_code = codes['color']  # B x 4  (binary or soft)
        color_embed_mu = embeds_mu['color']  # B x 32
        color_embed_logvar = embeds_logvar['color']  # B x 32
        shape_code = codes['shape']  # B x 4
        shape_embed_mu = embeds_mu['shape']  # B x 32
        shape_embed_logvar = embeds_logvar['shape']  # B x 32
        size_code = codes['size']  # B x 2
        size_embed_mu = embeds_mu['size']  # B x 32
        size_embed_logvar = embeds_logvar['size']  # B x 32
        camera_code = codes['camera']  # B x 3
        camera_embed_mu = embeds_mu['camera']  # B x 32
        camera_embed_logvar = embeds_logvar['camera']  # B x 32
        background_code = codes['background']  # B x 3
        background_embed_mu = embeds_mu['background']  # B x 32
        background_embed_logvar = embeds_logvar['background']  # B x 32
        horizontal_code = codes['horizontal']  # B x 40
        horizontal_embed_mu = embeds_mu['horizontal']  # B x 32
        horizontal_embed_logvar = embeds_logvar['horizontal']  # B x 32
        vertical_code = codes['vertical']  # B x 40
        vertical_embed_mu = embeds_mu['vertical']  # B x 32
        vertical_embed_logvar = embeds_logvar['vertical']  # B x 32

        color_mu = color_code.mm(self.mu['color'])  # B x 128
        color_logvar = color_code.mm(self.logvar['color'])  # B x 128
        color_embed = torch.cat((self.sampler(color_mu, color_logvar),
                                 self.sampler(color_embed_mu, color_embed_logvar)), 1)  # B x 160
        color_embed = self.input_color(color_embed)  # B x 256

        shape_mu = shape_code.mm(self.mu['shape'])  # B x 128
        shape_logvar = shape_code.mm(self.logvar['shape'])
        shape_embed = torch.cat((self.sampler(shape_mu, shape_logvar),
                                 self.sampler(shape_embed_mu, shape_embed_logvar)), 1)  # B x 160
        shape_embed = self.input_shape(shape_embed)  # B x 256

        size_mu = size_code.mm(self.mu['size'])
        size_logvar = size_code.mm(self.logvar['size'])
        size_embed = torch.cat((self.sampler(size_mu, size_logvar),
                                self.sampler(size_embed_mu, size_embed_logvar)), 1)  # B x 160
        size_embed = self.input_size(size_embed)  # B x 256

        col_shp_siz_embed = color_embed + shape_embed + size_embed  # B x 256
        col_shp_siz_embed = self.path_col_shp_siz(col_shp_siz_embed)  # B x 1024

        horizontal_mu = horizontal_code.mm(self.mu['horizontal'])
        horizontal_logvar = horizontal_code.mm(self.logvar['horizontal'])
        horizontal_embed = torch.cat((self.sampler(horizontal_mu, horizontal_logvar),
                                      self.sampler(horizontal_embed_mu, horizontal_embed_logvar)), 1)  # B x 160
        horizontal_embed = self.input_horizontal(horizontal_embed)

        vertical_mu = vertical_code.mm(self.mu['vertical'])
        vertical_logvar = vertical_code.mm(self.logvar['vertical'])
        vertical_embed = torch.cat((self.sampler(vertical_mu, vertical_logvar),
                                    self.sampler(vertical_embed_mu, vertical_embed_logvar)), 1)  # B x 160
        vertical_embed = self.input_vertical(vertical_embed)

        hor_ver_embed = horizontal_embed + vertical_embed
        hor_ver_embed = self.path_hor_ver(hor_ver_embed)  # B x 1024

        camera_mu = camera_code.mm(self.mu['camera'])
        camera_logvar = camera_code.mm(self.logvar['camera'])
        camera_embed = torch.cat((self.sampler(camera_mu, camera_logvar),
                                  self.sampler(camera_embed_mu, camera_embed_logvar)), 1)  # B x 160
        camera_embed = self.input_camera(camera_embed)  # B x 1024

        background_mu = background_code.mm(self.mu['background'])
        background_logvar = background_code.mm(self.logvar['background'])
        background_embed = torch.cat((self.sampler(background_mu, background_logvar),
                                      self.sampler(background_embed_mu, background_embed_logvar)), 1)  # B x 160
        background_embed = self.input_background(background_embed)  # B x 1024

        all_embed = col_shp_siz_embed + hor_ver_embed + camera_embed * 0.5 + background_embed * 0.5
        out_shallow = self.path_shallow(all_embed)
        out_deep = self.path_deep(all_embed)
        out = self.path_base(out_shallow + out_deep)  # B x 3 x 64 x 64
        return out


def code_transformer(codes):
    codes_prob, codes_oh_prob, codes_oh_trans, codes_oh_scale, codes_sg, codes_hg = {}, {}, {}, {}, {}, {}
    for name in codes:
        logits = codes[name]
        prob = torch.softmax(logits, 1)  # B x n_values

        argmax = torch.argmax(logits, 1, True) # B
        one_hot = torch.zeros_like(logits).scatter_(1, argmax, 1.0)
        one_hot_prob = prob * one_hot
        one_hot_trans = one_hot - one_hot_prob.detach() + one_hot_prob
        one_hot_scale = one_hot_prob / one_hot_prob.sum(1, keepdim=True).detach()

        soft_gumbel, hard_gumbel = U.gumbel_softmax(logits, tau=1, dim=1)

        codes_prob[name] = prob
        codes_oh_prob[name] = one_hot_prob
        codes_oh_trans[name] = one_hot_trans
        codes_oh_scale[name] = one_hot_scale
        codes_sg[name] = soft_gumbel
        codes_hg[name] = hard_gumbel

    return {'prob': codes_prob, 'one_hot_prob': codes_oh_prob, 'one_hot_trans': codes_oh_trans,
            'one_hot_scale': codes_oh_scale, 'soft_gumbel': codes_sg, 'hard_gumbel': codes_hg}
