import numpy as np
import math

def gaussian_kernel_1d(sigma, size):
    width = math.floor(size / 2)
    x = np.linspace(-width, width, size)

    c = 1 / (sigma * math.sqrt(2 * np.pi))
    x = c * np.exp(-1/2 * (x/sigma)**2)

    x = x / sum(x)

    return np.reshape(x, (size, 1))


def gaussian_kernel_2d(sigma, size):
    one_dim = gaussian_kernel_1d(sigma, size)

    return np.outer(one_dim, one_dim)


def gaussian_pos_embedding(size, sigma):
    img = gaussian_kernel_2d(sigma, size)

    return img

