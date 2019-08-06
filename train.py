from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from data import Mpi3dToy
from archs import (Encoder, Decoder, code_transformer, preprocess, CodeCompressor, Discriminator,
                   neg_entropy_regularizer, total_correction_regularizer, per_sample_entropy_regularizer)


CODE_OUT_MODE = 'hard_gumbel_trans'
USE_DISCRIMINATOR = False

factors = {'color': 8, 'shape': 10, 'size': 4, 'camera': 6, 'background': 5, 'horizontal': 16, 'vertical': 20}
dataset = Mpi3dToy()


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.encoder = Encoder(factors)
        self.decoder = Decoder(factors)

    def forward(self, images, step, max_steps, encoding_only=False):
        codes_logits, embeds_mu, embeds_logvar = self.encoder(images)
        transformed_codes = code_transformer(codes_logits, step, max_steps)

        if encoding_only:
            return transformed_codes
        else:
            recon_images = self.decoder(transformed_codes[CODE_OUT_MODE], embeds_mu, embeds_logvar)
            return recon_images, transformed_codes, self.encoder.regularizer(embeds_mu, embeds_logvar)


class CompressNet(nn.Module):
    def __init__(self):
        super(CompressNet, self).__init__()
        self.compressors = nn.ModuleDict({name: CodeCompressor(n_values) for name, n_values in factors.items()})

    def forward(self, transformed_codes):
        z_factors = {}
        loss = 0
        for name, oh_code in transformed_codes[CODE_OUT_MODE].items():
            prob_code = transformed_codes['prob'][name]
            loss_oh, z, _ = self.compressors[name](oh_code.detach(), prob_code.detach())
            z_factors[name] = z
            loss = loss + loss_oh
        return loss, z_factors


class DiscrNet(nn.Module):
    def __init__(self):
        super(DiscrNet, self).__init__()
        self.discriminators_1 = nn.ModuleDict({name: Discriminator(n_values) for name, n_values in factors.items()})

    def forward(self, transformed_codes, mode):
        ret = {}
        trans_codes = transformed_codes['prob']
        for name, oh_code in trans_codes.items():
            ret[name] = self.discriminators_1[name].train_loss(oh_code.detach()) if mode == 'train' \
                else self.discriminators_1[name].get_score(oh_code)
        return ret


def reg_net(transformed_codes, step):
    trans_codes = transformed_codes[CODE_OUT_MODE]
    neg_entropy_reg = {}
    for name, n_values in factors.items():
        neg_entropy_reg[name] = neg_entropy_regularizer(trans_codes[name])
        neg_entropy_reg['total'] = neg_entropy_reg[name] if 'total' not in neg_entropy_reg \
            else neg_entropy_reg['total'] + neg_entropy_reg[name]

    codes_prob = transformed_codes['prob']
    codes_logprob = transformed_codes['logprob']
    per_sample_entropy_reg = {}
    for name, n_values in factors.items():
        per_sample_entropy_reg[name] = per_sample_entropy_regularizer(codes_prob[name], codes_logprob[name])
        per_sample_entropy_reg['total'] = per_sample_entropy_reg[name] if 'total' not in per_sample_entropy_reg \
            else per_sample_entropy_reg['total'] + per_sample_entropy_reg[name]

    trans_codes = transformed_codes[CODE_OUT_MODE]
    tc_reg = {}
    names = list(factors.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name_i = names[i]
            name_j = names[j]
            tc = total_correction_regularizer(trans_codes[name_i], trans_codes[name_j])
            tc_reg['(%s,%s)' % (name_i, name_j)] = tc
            tc_reg['total'] = tc if 'total' not in tc_reg else tc_reg['total'] + tc

    if step % 100 == 0:
        print('[Step %d][%s] neg_entropy_reg:' % (step, CODE_OUT_MODE))
        print('  %s' % ', '.join(['%s: %.4f' % (name, val.cpu().item()) for name, val in neg_entropy_reg.items()]))
        print('[Step %d][%s] per_sample_entropy_reg:' % (step, 'prob'))
        print('  %s' % ', '.join(['%s: %.4f' % (name, val.cpu().item()) for name, val in per_sample_entropy_reg.items()]))
        print('[Step %d][%s] tc_reg:' % (step, CODE_OUT_MODE))
        print('  %s' % ', '.join(['%s: %.3f' % (name, val.cpu().item()) for name, val in tc_reg.items()]))

    return neg_entropy_reg['total'], per_sample_entropy_reg['total'], tc_reg['total']


def monitor(transformed_codes, step):
    with torch.no_grad():
        trans_codes = transformed_codes[CODE_OUT_MODE]
        neg_entropy_reg = {}
        for name, n_values in factors.items():
            neg_entropy_reg[name] = neg_entropy_regularizer(trans_codes[name])
            neg_entropy_reg['total'] = neg_entropy_reg[name] if 'total' not in neg_entropy_reg \
                else neg_entropy_reg['total'] + neg_entropy_reg[name]

        tc_reg = {}
        names = list(factors.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_i = names[i]
                name_j = names[j]
                tc = total_correction_regularizer(trans_codes[name_i], trans_codes[name_j])
                tc_reg['(%s,%s)' % (name_i, name_j)] = tc
                tc_reg['total'] = tc if 'total' not in tc_reg else tc_reg['total'] + tc

        if step % 100 == 0:
            print('[Step %d][%s] monitor neg_entropy_reg:' % (step, CODE_OUT_MODE))
            print('  %s' % ', '.join(['%s: %.4f' % (name, val.cpu().item()) for name, val in neg_entropy_reg.items()]))
            print('[Step %d][%s] monitor tc_reg:' % (step, CODE_OUT_MODE))
            print('  %s' % ', '.join(['%s: %.3f' % (name, val.cpu().item()) for name, val in tc_reg.items()]))


main_net = MainNet().cuda()
main_optimizer = optim.Adam(main_net.parameters(), lr=0.0002, betas=(0.5, 0.9), weight_decay=0.00001)

compress_net = CompressNet().cuda()
compress_optimizer = optim.Adam(compress_net.parameters(), lr=0.0001)

discr_net = DiscrNet().cuda()
discr_optimizer = optim.Adam(discr_net.parameters(), lr=0.0001, betas=(0, 0.9))

batch_size = 256
max_steps = 100000

code_stats = {name: torch.zeros(n_values).cuda() for name, n_values in factors.items()}


def train_discriminators(step):
    losses_trace = defaultdict(list)
    for _ in range(2):
        _, raw_images = dataset.get_samples(batch_size)
        images = preprocess(raw_images).cuda()
        transformed_codes = main_net(images, step, max_steps, encoding_only=True)

        loss_sum = 0
        discr_losses = discr_net(transformed_codes, 'train')
        for name, loss in discr_losses.items():
            losses_trace[name].append(loss.detach())
            loss_sum = loss_sum + loss

        discr_optimizer.zero_grad()
        loss_sum.backward()
        discr_optimizer.step()

    if step % 100 == 0:
        print('[Step %d] train_discriminators:' % step)
        for name in losses_trace:
            print('  %s: %s' % (name, ', '.join(map(lambda x: '%.4f' % x.cpu().item(), losses_trace[name]))))


def get_discriminators_scores(transformed_codes, step):
    score_sum = 0
    score_diffs = {}
    discr_scores = discr_net(transformed_codes, 'score')
    for name, (score, score_diff) in discr_scores.items():
        score_diffs[name] = score_diff.detach()
        score_sum = score_sum + score

    if step % 100 == 0:
        print('[Step %d] discriminators_scores:' % step)
        print('  %s' % ', '.join(['%s: %.4f' % (name, diff.cpu().item()) for name, diff in score_diffs.items()]))
    return score_sum


def train():
    main_net.train()
    for step in range(max_steps):
        if USE_DISCRIMINATOR:
            train_discriminators(step)

        _, raw_images = dataset.get_samples(batch_size)
        images = preprocess(raw_images).cuda()
        recon_images, transformed_codes, enc_reg_loss = main_net(images, step, max_steps)

        images = images.reshape(-1, 3 * 64 * 64)
        recon_images = recon_images.reshape(-1, 3 * 64 * 64)
        recon_loss = F.mse_loss(recon_images, images, reduction='none').sum(1).mean()

        discr_score = get_discriminators_scores(transformed_codes, step) if USE_DISCRIMINATOR else 0

        neg_entropy_reg, per_sample_entropy_reg, tc_reg = reg_net(transformed_codes, step)
        #monitor(transformed_codes, step)

        beta1 = np.random.rand() * 20
        beta2 = np.random.rand() * 20
        beta3 = np.random.rand() * 40
        beta4 = np.random.rand() * 40
        loss = recon_loss + beta1 * enc_reg_loss - discr_score + beta2 * neg_entropy_reg + beta3 * per_sample_entropy_reg + beta4 * tc_reg
        main_optimizer.zero_grad()
        loss.backward()
        main_optimizer.step()

        comp_loss, z_factors = compress_net(transformed_codes)

        compress_optimizer.zero_grad()
        comp_loss.backward()
        compress_optimizer.step()

        for name, oh_code in transformed_codes[CODE_OUT_MODE].items():
            code_stats[name] = code_stats[name] * 0.9 + oh_code.detach().mean(0) * 0.1

        if step % 100 == 0:
            print('[Step %d] loss: %.4f, recon_loss: %.4f, enc_reg_loss: %.4f, neg_discr_score: %.4f, '
                  'neg_entropy_reg: %.4f, per_sample_entropy_reg: %.4f, tc_reg: %.4f, comp_loss: %.4f' %
                  (step, loss, recon_loss, enc_reg_loss, - discr_score, neg_entropy_reg, per_sample_entropy_reg, tc_reg, comp_loss))
            print('  %s' % ', '.join(('%s: (%s)' % (k, ','.join(map(lambda x: '%.2f' % x, v.tolist()))) for k, v in code_stats.items())))
            print()

        if torch.isnan(recon_loss).any().cpu().item() == 1:
            debug_checkpoint = True


train()

main_net.eval()
factors, raw_images = dataset.get_samples(batch_size)
images = preprocess(raw_images).cuda()
recon_images, transformed_codes, _ = main_net(images, 1, 1)

images = images.sigmoid().detach().cpu().numpy().transpose([0, 2, 3, 1])
recon_images = recon_images.sigmoid().detach().cpu().numpy().transpose([0, 2, 3, 1])
dataset.show_images(5, 5, factors, raw_images, 'original_images')
dataset.show_images(5, 5, factors, images, 'processed_images')
dataset.show_images(5, 5, factors, recon_images, 'predicated_images')


from disentanglement_lib.data.ground_truth import mpi3d
from disentanglement_lib.evaluation.metrics import dci, factor_vae, sap_score, mig, irs, utils


def representation_fn(x):
    main_net.eval()
    with torch.no_grad():
        images = preprocess(x).cuda()
        transformed_codes = main_net(images, 1, 1, encoding_only=True)
        comp_loss, z_factors = compress_net(transformed_codes)
        repr_code = torch.cat([z for k, z in z_factors.items()], 1).cpu().numpy()
    return repr_code

ground_truth_data = mpi3d.MPI3D()

print('representation_fn')
scores = factor_vae.compute_factor_vae(ground_truth_data, representation_fn, np.random.RandomState(0), 64, 10000, 5000, 10000)
print('  factor_vae: %.6f' % scores['eval_accuracy'])

scores = dci.compute_dci(ground_truth_data, representation_fn, np.random.RandomState(0), 10000, 5000)
print('  dci: %.6f' % scores['disentanglement'])

scores = sap_score.compute_sap(ground_truth_data, representation_fn, np.random.RandomState(0), 10000, 5000, continuous_factors=False)
print('  sap_score: %.6f' % scores['SAP_score'])

import gin.tf
gin.bind_parameter("discretizer.discretizer_fn", utils._histogram_discretize)
gin.bind_parameter("discretizer.num_bins", 20)

scores = mig.compute_mig(ground_truth_data, representation_fn, np.random.RandomState(0), 10000)
print('  mig: %.6f' % scores['discrete_mig'])

gin.bind_parameter("irs.batch_size", 16)
scores = irs.compute_irs(ground_truth_data, representation_fn, np.random.RandomState(0), num_train=10000)
print('  irs: %.6f' % scores['IRS'])
