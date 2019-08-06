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
        self.output_color = nn.ModuleDict({'code': nn.Linear(256, self.factors['color']),
                                           'embed': nn.Linear(256, 64)})  # 1. force to follow N(0, 1) 2. penalty for large values
        self.output_shape = nn.ModuleDict({'code': nn.Linear(256, self.factors['shape']),
                                           'embed': nn.Linear(256, 64)})
        self.output_size = nn.ModuleDict({'code': nn.Linear(256, self.factors['size']),
                                          'embed': nn.Linear(256, 64)})

        self.weight_hor_ver = [1.36, 0.78, 0.67]
        self.path_hor_ver_shallow = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.path_hor_ver_deep = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                               nn.Linear(512, 256), nn.ReLU())
        self.path_hor_ver_deeper = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                                 nn.Linear(512, 256), nn.ReLU(),
                                                 nn.Linear(256, 256), nn.ReLU())

        self.weight_hor = [0.24, 0.0, 0.14]
        self.weight_ver = [0.30, 0.58, 0.0]
        self.output_horizontal = nn.ModuleDict({'code': nn.Linear(256, self.factors['horizontal']),
                                                'embed': nn.Linear(256, 64)})
        self.output_vertical = nn.ModuleDict({'code': nn.Linear(256, self.factors['vertical']),
                                              'embed': nn.Linear(256, 64)})

        self.weight_cam = [1.35, 1.15, 0.56]
        self.weight_bkg = [1.11, 0.0, 1.54]
        self.output_camera = nn.ModuleDict({'code': nn.Linear(1024, self.factors['camera']),
                                            'embed': nn.Linear(1024, 64)})
        self.output_background = nn.ModuleDict({'code': nn.Linear(1024, self.factors['background']),
                                                'embed': nn.Linear(1024, 64)})

    def regularizer(self, embeds):
        reg_loss = 0
        for name in embeds:
            reg_loss += embeds[name].pow(2).sum(1).mean()
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
        color_code = self.output_color['code'](in_color)  # B x 4
        color_embed = self.output_color['embed'](in_color)  # B x 32

        weight = self.weight_shp
        in_shape = out_css_shallow * weight[0] + out_css_deep * weight[1] + out_css_deeper * weight[2]
        shape_code = self.output_shape['code'](in_shape)  # B x 4
        shape_embed = self.output_shape['embed'](in_shape)  # B x 32

        weight = self.weight_siz
        in_size = out_css_shallow * weight[0] + out_css_deep * weight[1] + out_css_deeper * weight[2]
        size_code = self.output_size['code'](in_size)  # B x 4
        size_embed = self.output_size['embed'](in_size)  # B x 32

        weight = self.weight_hor_ver
        in_hv = out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2]
        out_hv_shallow = self.path_hor_ver_shallow(in_hv)
        out_hv_deep = self.path_hor_ver_deep(in_hv)
        out_hv_deeper = self.path_hor_ver_deeper(in_hv)

        weight = self.weight_hor
        in_horizontal = out_hv_shallow * weight[0] + out_hv_deep * weight[1] + out_hv_deeper * weight[2]
        horizontal_code = self.output_horizontal['code'](in_horizontal)
        horizontal_embed = self.output_horizontal['embed'](in_horizontal)

        weight = self.weight_ver
        in_vertical = out_hv_shallow * weight[0] + out_hv_deep * weight[1] + out_hv_deeper * weight[2]
        vertical_code = self.output_vertical['code'](in_vertical)
        vertical_embed = self.output_vertical['embed'](in_vertical)

        weight = self.weight_cam
        in_camera = out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2]
        camera_code = self.output_camera['code'](in_camera)
        camera_embed = self.output_camera['embed'](in_camera)

        weight = self.weight_bkg
        in_background = out_shallow * weight[0] + out_deep * weight[1] + out_pool * weight[2]
        background_code = self.output_background['code'](in_background)
        background_embed = self.output_background['embed'](in_background)

        codes = {'color': color_code, 'shape': shape_code, 'size': size_code,
                 'camera': camera_code, 'background': background_code,
                 'horizontal': horizontal_code, 'vertical': vertical_code}
        embeds = {'color': color_embed, 'shape': shape_embed, 'size': size_embed,
                  'camera': camera_embed, 'background': background_embed,
                  'horizontal': horizontal_embed, 'vertical': vertical_embed}
        return codes, embeds


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.factors = {'color': 4, 'shape': 4, 'size': 2, 'camera': 3, 'background': 3, 'horizontal': 40, 'vertical': 40}
        self.mu = nn.ParameterDict({
            'color': nn.Parameter(torch.randn(self.factors['color'], 64)),  # 4 x 64, no regularization
            'shape': nn.Parameter(torch.randn(self.factors['shape'], 64)),
            'size': nn.Parameter(torch.randn(self.factors['size'], 64)),
            'camera': nn.Parameter(torch.randn(self.factors['camera'], 64)),
            'background': nn.Parameter(torch.randn(self.factors['background'], 64)),
            'horizontal': nn.Parameter(torch.randn(self.factors['horizontal'], 64)),
            'vertical': nn.Parameter(torch.randn(self.factors['vertical'], 64))
        })
        self.logvar = nn.ParameterDict({
            'color': nn.Parameter(torch.zeros(self.factors['color'], 64)),  # 4 x 64, penalty for large values
            'shape': nn.Parameter(torch.zeros(self.factors['shape'], 64)),
            'size': nn.Parameter(torch.zeros(self.factors['size'], 64)),
            'camera': nn.Parameter(torch.zeros(self.factors['camera'], 64)),
            'background': nn.Parameter(torch.zeros(self.factors['background'], 64)),
            'horizontal': nn.Parameter(torch.zeros(self.factors['horizontal'], 64)),
            'vertical': nn.Parameter(torch.zeros(self.factors['vertical'], 64))
        })

        self.input_color = nn.Linear(64, 256)
        self.input_shape = nn.Linear(64, 256)
        self.input_size = nn.Linear(64, 256)
        self.path_col_shp_siz = nn.Sequential(nn.ReLU(), nn.Linear(256, 1024))

        self.input_horizontal = nn.Linear(64, 256)
        self.input_vertical = nn.Linear(64, 256)
        self.path_hor_ver = nn.Sequential(nn.ReLU(), nn.Linear(256, 1024))

        self.input_camera = nn.Linear(64, 1024)
        self.input_background = nn.Linear(64, 1024)

        self.path_shallow = nn.Sequential(nn.ReLU(), nn.Linear(1024, 1024),
                                          U.Lambda(lambda x: x.reshape(-1, 16, 8, 8)),  # 16 x 8 x 8
                                          nn.ReLU(), U.deconv2d(16, 64, 1, 1, True, 8))  # 64 x 8 x 8
        self.path_deep = nn.Sequential(nn.ReLU(), nn.Linear(1024, 1024),
                                       U.Lambda(lambda x: x.reshape(-1, 64, 4, 4)),  # 64 x 4 x 4
                                       nn.ReLU(), U.deconv2d(64, 64, 4, 1, True, 4),  # 64 x 4 x 4
                                       nn.ReLU(), U.deconv2d(64, 64, 4, 2, True, 8))  # 64 x 8 x 8
        self.path_base = nn.Sequential(nn.ReLU(), U.deconv2d(64, 16, 4, 2, True, 16),  # 16 x 16 x 16
                                       nn.ReLU(), U.deconv2d(16, 3, 6, 4, True, 64))  # 3 x 64 x 64
        self.alpha = 1.0

    def regularizer(self):
        reg_loss = 0
        for name in self.logvar:
            reg_loss += torch.exp(self.logvar[name]).sum(1).mean()
        return reg_loss

    def forward(self, codes, embeds):
        color_code = codes['color']  # B x 4  (binary or soft)
        color_embed = embeds['color']  # B x 64
        shape_code = codes['shape']  # B x 4
        shape_embed = embeds['shape']  # B x 64
        size_code = codes['size']  # B x 2
        size_embed = embeds['size']  # B x 64
        camera_code = codes['camera']  # B x 3
        camera_embed = embeds['camera']  # B x 64
        background_code = codes['background']  # B x 3
        background_embed = embeds['background']  # B x 64
        horizontal_code = codes['horizontal']  # B x 40
        horizontal_embed = embeds['horizontal']  # B x 64
        vertical_code = codes['vertical']  # B x 40
        vertical_embed = embeds['vertical']  # B x 64

        color_mu = color_code.mm(self.mu['color'])
        color_logvar = color_code.mm(self.logvar['color'])
        color_embed = color_mu + color_embed * torch.exp(color_logvar / 2)  # B x 64
        color_embed = self.input_color(color_embed)  # B x 256

        shape_mu = shape_code.mm(self.mu['shape'])
        shape_logvar = shape_code.mm(self.logvar['shape'])
        shape_embed = shape_mu + shape_embed * torch.exp(shape_logvar / 2)  # B x 64
        shape_embed = self.input_shape(shape_embed)  # B x 256

        size_mu = size_code.mm(self.mu['size'])
        size_logvar = size_code.mm(self.logvar['size'])
        size_embed = size_mu + size_embed * torch.exp(size_logvar / 2)  # B x 64
        size_embed = self.input_size(size_embed)  # B x 256

        col_shp_siz_embed = color_embed + shape_embed + size_embed  # B x 256
        col_shp_siz_embed = self.path_col_shp_siz(col_shp_siz_embed)  # B x 1024

        horizontal_mu = horizontal_code.mm(self.mu['horizontal'])
        horizontal_logvar = horizontal_code.mm(self.logvar['horizontal'])
        horizontal_embed = horizontal_mu + horizontal_embed * torch.exp(horizontal_logvar / 2)
        horizontal_embed = self.input_horizontal(horizontal_embed)

        vertical_mu = vertical_code.mm(self.mu['vertical'])
        vertical_logvar = vertical_code.mm(self.logvar['vertical'])
        vertical_embed = vertical_mu + vertical_embed * torch.exp(vertical_logvar / 2)
        vertical_embed = self.input_vertical(vertical_embed)

        hor_ver_embed = horizontal_embed + vertical_embed
        hor_ver_embed = self.path_hor_ver(hor_ver_embed)  # B x 1024

        camera_mu = camera_code.mm(self.mu['camera'])
        camera_logvar = camera_code.mm(self.logvar['camera'])
        camera_embed = camera_mu + camera_embed * torch.exp(camera_logvar / 2)
        camera_embed = self.input_camera(camera_embed)  # B x 1024

        background_mu = background_code.mm(self.mu['background'])
        background_logvar = background_code.mm(self.logvar['background'])
        background_embed = background_mu + background_embed * torch.exp(background_logvar / 2)
        background_embed = self.input_background(background_embed)  # B x 1024

        all_embed = col_shp_siz_embed + hor_ver_embed + camera_embed * 0.5 + background_embed * 0.5
        out_shallow = self.path_shallow(all_embed)
        out_deep = self.path_deep(all_embed)
        out = self.path_base(out_shallow + out_deep)  # B x 3 x 64 x 64
        return out * self.alpha


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
