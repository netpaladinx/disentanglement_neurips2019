from itertools import chain

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from data import Mpi3dToy
from archs import Encoder, Decoder, code_transformer, preprocess, CodeCompressor


dataset = Mpi3dToy()
encoder = Encoder().cuda()
decoder = Decoder().cuda()
optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=0.0001)

code_compressors = {name: CodeCompressor(n_values).cuda() for name, n_values in dataset.factors.items()}
compressors_optimizer = optim.Adam(chain(*[code_compressors[name].parameters() for name in code_compressors]), lr=0.0001)

batch_size = 128
max_steps = 10000


def train():
    global images_center
    global count

    decoder.train()
    for step in range(max_steps):
        factors, raw_images = dataset.get_samples(batch_size)
        images = preprocess(raw_images).cuda()

        codes_logits, embeds_mu, embeds_logvar = encoder(images)
        transformed_codes = code_transformer(codes_logits)
        recon_images = decoder(transformed_codes['one_hot_scale'], embeds_mu, embeds_logvar)

        images = images.reshape(-1, 3 * 64 * 64)
        recon_images = recon_images.reshape(-1, 3 * 64 * 64)
        recon_loss = F.mse_loss(recon_images, images, reduction='none').sum(1).mean()
        enc_reg_loss = encoder.regularizer(embeds_mu, embeds_logvar)
        loss = recon_loss + enc_reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('[Step %d] loss: %.4f, recon_loss: %.4f, enc_reg_loss: %.4f' %
                  (step, loss, recon_loss, enc_reg_loss))
            #print('  %s' % ', '.join(('%s: %s' % (k, str(v.mean(0))) for k, v in transformed_codes['one_hot'].items())))

        z_factors = {}
        loss = 0
        for name, oh_code in transformed_codes['one_hot'].items():
            loss_oh, z, _ = code_compressors[name](oh_code.detach())
            z_factors[name] = z
            loss = loss + loss_oh
        compressors_optimizer.zero_grad()
        loss.backward()
        compressors_optimizer.step()

        if step % 100 == 0:
            print('[step %d] loss: %.4f' % (step, loss))


train()

# decoder.eval()
# factors, raw_images = dataset.get_samples(batch_size)
# images = preprocess(raw_images).cuda()
# codes_logits, embeds_mu, embeds_logvar = encoder(images)
# transformed_codes = code_transformer(codes_logits)
# recon_images = decoder(transformed_codes['one_hot_scale'], embeds_mu, embeds_logvar)
#
# images = images.sigmoid().detach().cpu().numpy().transpose([0, 2, 3, 1])
# recon_images = recon_images.sigmoid().detach().cpu().numpy().transpose([0, 2, 3, 1])
# dataset.show_images(5, 5, factors, raw_images, 'original_images')
# dataset.show_images(5, 5, factors, images, 'processed_images')
# dataset.show_images(5, 5, factors, recon_images, 'predicated_images')


from disentanglement_lib.data.ground_truth import mpi3d
from disentanglement_lib.evaluation.metrics import dci, factor_vae, sap_score, mig, irs, utils

def representation_fn(x):
    with torch.no_grad():
        images = preprocess(x).cuda()
        codes_logits, _, _ = encoder(images)
        transformed_codes = code_transformer(codes_logits)

        z_factors = {}
        for name, oh_code in transformed_codes['one_hot'].items():
            _, z, _ = code_compressors[name](oh_code)
            z_factors[name] = z

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
