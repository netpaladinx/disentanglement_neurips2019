import torch
import torch.nn as nn


def get_padding(kernel_size, stride=None, h_or_w=None):
    unused = (h_or_w - 1) % stride if h_or_w and stride else 0
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg - unused
    return (pad_beg, pad_end, pad_beg, pad_end)


def slice2d(x, padding):
    pad_t, pad_b, pad_l, pad_r = padding
    return x.narrow(2, pad_t, x.size(2) - pad_t - pad_b).narrow(3, pad_l, x.size(3) - pad_l - pad_r)


class Lambda(nn.Module):
    def __init__(self, fn, *args):
        super(Lambda, self).__init__()
        self.args = args
        self.fn = fn

    def forward(self, x):
        return self.fn(x, *self.args)


def conv2d(in_channels, out_channels, kernel_size, stride, bias, in_h_or_w=None):
    if kernel_size > 1:
        return nn.Sequential(
            nn.ZeroPad2d(get_padding(kernel_size, stride, in_h_or_w)),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias))
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)


def deconv2d(in_channels, out_channels, kernel_size, stride, bias, out_h_or_w=None):
    if kernel_size > 1:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias),
            Lambda(slice2d, get_padding(kernel_size, stride, out_h_or_w)))
    else:
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)


def gumbel_softmax(logits, tau=1, dim=-1):
    gumbels = - torch.log(torch.empty_like(logits).exponential_() + 1e-20)  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # Reparametrization trick.
    y_soft = gumbels.softmax(dim)

    if torch.isnan(y_soft).any().cpu().item() == 1:
        debug_checkpoint = True

    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    y_hard_trans = y_hard - y_soft.detach() + y_soft
    y_hard_scale = y_hard / y_soft.detach() * y_soft
    return y_soft, y_hard_trans, y_hard_scale
