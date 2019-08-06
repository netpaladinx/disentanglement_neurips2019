import numpy as np
import matplotlib.pyplot as plt

import torch

class Mpi3dToy(object):
    def __init__(self):
        self.data = np.load('mpi3d_toy/mpi3d_toy.npz')['images'].reshape([4, 4, 2, 3, 3, 40, 40, 64, 64, 3])
        self.factors = {'color': 4, 'shape': 4, 'size': 2, 'camera': 3, 'background': 3, 'horizontal': 40, 'vertical': 40}

    def get_samples(self, n_samples):
        sampled_factors = tuple([np.random.randint(0, self.factors[name], n_samples)
                                 for name in ('color', 'shape', 'size', 'camera', 'background', 'horizontal', 'vertical')])
        sampled_images = self.data[sampled_factors]
        sampled_images = np.array(sampled_images, dtype=np.float32) / 255
        sampled_factors = np.stack(sampled_factors, axis=1)
        return sampled_factors, sampled_images

    def show_images(self, nr=5, nc=5, factors=None, images=None, name=None):
        n_samples = nr * nc
        if factors is None or images is None:
            factors, images = self.get_samples(n_samples)
        else:
            factors = factors[:n_samples]
            images = images[:n_samples]
        fig, axs = plt.subplots(nr, nc, figsize=(20, 10))
        for ax, fac, img in zip(axs.ravel(), factors, images):
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(str(fac), fontsize=10)
        plt.show()
        fig.savefig('mpi3dtoy_images.pdf' if name is None else '%s.pdf' % name)