from matplotlib import cm
from optimizers import Optimizer
from losses import compute_loss
from mpl_toolkits.mplot3d import Axes3D

import pylab as plt
import numpy as np
import torch

def init_plot():
    fig = plt.figure(figsize=(3, 2), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    params = {'legend.fontsize': 3,
              'legend.handlelength': 3}
    plt.rcParams.update(params)
    plt.axis('off')
    # visualize cost function as a contour plot
    w1_val = w2_val = np.arange(-1.5, 1.5, 0.005, dtype=np.float32)
    w1_val_mesh, w2_val_mesh = np.meshgrid(w1_val, w2_val)
    w1_val_mesh_flat = w1_val_mesh.reshape([-1, 1])
    w2_val_mesh_flat = w2_val_mesh.reshape([-1, 1])

    loss_val_mesh_flat = compute_loss(w1=torch.FloatTensor(w1_val_mesh_flat),
                                      w2=torch.FloatTensor(w2_val_mesh_flat)).numpy()
    loss_val_mesh = loss_val_mesh_flat.reshape(w1_val_mesh.shape)
    levels = np.arange(-10, 1, 0.05)

    ax.plot_surface(w1_val_mesh, w2_val_mesh, loss_val_mesh, alpha=.4, cmap=cm.coolwarm)

    plt.draw()
    # 3d plot camera zoom, angle
    xlm = ax.get_xlim3d()
    ylm = ax.get_ylim3d()
    zlm = ax.get_zlim3d()
    ax.set_xlim3d(xlm[0] * 0.5, xlm[1] * 0.5)
    ax.set_ylim3d(ylm[0] * 0.5, ylm[1] * 0.5)
    ax.set_zlim3d(zlm[0] * 0.5, zlm[1] * 0.5)
    azm = ax.azim
    ele = ax.elev + 40
    ax.view_init(elev=ele, azim=azm)

    return ax

def main():
    optimizer_list = [Optimizer(name) for name in ["Adam",
                                           "Adagrad",
                                           "RMSprop",
                                           "SGD"]]

    # use last location to draw a line to the current location
    for i in range(1000):
        for j in range(len(optimizer_list)):
            model = optimizer_list[j]
            model.train_step()

        if i % 10 == 0:
            ax = init_plot()
            for j in range(len(optimizer_list)):
                optimizer_list[j].plot(ax)
            plt.legend()
            plt.show()
    # plt.savefig('figures/' + str(iter) + '.png')
    # print('iteration: {}'.format(iter))
    #
    # plt.pause(0.0001)

main()
