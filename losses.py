import torch

import numpy as np

# cost function
def compute_loss(w1=None, w2=None):
    # two local minima near (0, 0)
    #     z = __f1(x, y)

    # 3rd local minimum at (-0.5, -0.8)
    z = -1 * f2(w1, w2, w1_mean=-0.5, w2_mean=-0.8, w1_sig=0.35, w2_sig=0.35)

    # one steep gaussian trench at (0, 0)
    #     z -= __f2(x, y, w1_mean=0, w2_mean=0, w1_sig=0.2, w2_sig=0.2)

    # three steep gaussian trenches
    z -= f2(w1, w2, w1_mean=1.0, w2_mean=-0.5, w1_sig=0.2, w2_sig=0.2)
    z -= f2(w1, w2, w1_mean=-1.0, w2_mean=0.5, w1_sig=0.2, w2_sig=0.2)
    z -= f2(w1, w2, w1_mean=-0.5, w2_mean=-0.8, w1_sig=0.2, w2_sig=0.2)

    return z


# noisy hills of the cost function
def __f1(x, y):
    return -1 * torch.sin(x * x) * torch.cos(3 * y * y) * torch.exp(-(x * y) * (x * y)) - torch.exp(-(x + y) * (x + y))


# bivar gaussian hills of the cost function
def f2(w1, w2, w1_mean, w2_mean, w1_sig, w2_sig):
    normalizing = 1 / (2 * np.pi * w1_sig * w2_sig)
    w1_exp = (-1 * (w1 - w1_mean)**2) / (2 * np.square(w1_sig))
    w2_exp = (-1 * (w2 - w2_mean)**2) / (2 * np.square(w2_sig))
    return normalizing * torch.exp(w1_exp + w2_exp)

