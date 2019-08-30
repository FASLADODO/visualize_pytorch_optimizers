from matplotlib import cm
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
    model_list = [MODEL(name) for name in ["Adam",
                                           "Adagrad",
                                           "RMSprop",
                                           "SGD"]]

    # use last location to draw a line to the current location
    for i in range(1000):
        for j in range(len(model_list)):
            model = model_list[j]
            model.train_step()

        if i % 10 == 0:
            ax = init_plot()
            for j in range(len(model_list)):
                model_list[j].plot(ax)
            plt.legend()
            plt.show()
    # plt.savefig('figures/' + str(iter) + '.png')
    # print('iteration: {}'.format(iter))
    #
    # plt.pause(0.0001)

class MODEL:
    def __init__(self, name):
        self.w1 = torch.nn.Parameter(torch.FloatTensor([0.75]))
        self.w2 = torch.nn.Parameter(torch.FloatTensor([1.0]))
        self.name = name

        W = [self.w1, self.w2]

        if name == "Adam":
            self.opt = torch.optim.Adam(W, lr=1e-1)
            self.color = "r"
        elif name == "Adagrad":
            self.opt = torch.optim.Adagrad(W, lr=1e-1)
            self.color = "b"
        elif name == "RMSprop":
            self.opt = torch.optim.RMSprop(W)
            self.color = "g"
        elif name == "SGD":
            self.opt = torch.optim.SGD(W, lr=100)
            self.color = "y"
        self.w1_list = []
        self.w2_list = []
        self.loss_list = []

    def train_step(self):
        self.update()

        self.opt.zero_grad()
        loss = compute_loss(self.w1, self.w2)
        loss.backward()
        self.opt.step()

        return loss

    def update(self):
        self.w1_list += [float(self.w1)]
        self.w2_list += [float(self.w2)]
        self.loss_list += [float(compute_loss(self.w1, self.w2))]

    def plot(self, ax):
        ax.plot(self.w1_list,
                self.w2_list,
                self.loss_list,
                linewidth=0.5,
                label=self.name,
                color=self.color)
        ax.scatter(self.w1_list[-1],
                   self.w2_list[-1],
                   self.loss_list[-1],
                   s=3, depthshade=True,
                   label=self.name,
                   color=self.color)


# # create variable pair (x, y) for each optimizer
# x_var, y_var = [], []
# for i in range(7):
#     x_var.append(tf.Variable(x_i, [1], dtype=tf.float32))
#     y_var.append(tf.Variable(y_i, [1], dtype=tf.float32))
#
# # create separate graph for each variable pairs
# cost = []
# for i in range(7):
#     cost.append(cost_func(x_var[i], y_var[i])[2])
#
# # define method of gradient descent for each graph
# # optimizer label name, learning rate, color
# ops_param = np.array([['Adadelta', 50.0, 'b'],
#                      ['Adagrad', 0.10, 'g'],
#                      ['Adam', 0.05, 'r'],
#                      ['Ftrl', 0.5, 'c'],
#                      ['GD', 0.05, 'm'],
#                      ['Momentum', 0.01, 'y'],
#                      ['RMSProp', 0.02, 'k']])
#
# ops = []
# ops.append(tf.train.AdadeltaOptimizer(float(ops_param[0, 1])).minimize(cost[0]))
# ops.append(tf.train.AdagradOptimizer(float(ops_param[1, 1])).minimize(cost[1]))
# ops.append(tf.train.AdamOptimizer(float(ops_param[2, 1])).minimize(cost[2]))
# ops.append(tf.train.FtrlOptimizer(float(ops_param[3, 1])).minimize(cost[3]))
# ops.append(tf.train.GradientDescentOptimizer(float(ops_param[4, 1])).minimize(cost[4]))
# ops.append(tf.train.MomentumOptimizer(float(ops_param[5, 1]), momentum=0.95).minimize(cost[5]))
# ops.append(tf.train.RMSPropOptimizer(float(ops_param[6, 1])).minimize(cost[6]))
#
# # 3d plot camera zoom, angle
# xlm = ax.get_xlim3d()
# ylm = ax.get_ylim3d()
# zlm = ax.get_zlim3d()
# ax.set_xlim3d(xlm[0] * 0.5, xlm[1] * 0.5)
# ax.set_ylim3d(ylm[0] * 0.5, ylm[1] * 0.5)
# ax.set_zlim3d(zlm[0] * 0.5, zlm[1] * 0.5)
# azm = ax.azim
# ele = ax.elev + 40
# ax.view_init(elev=ele, azim=azm)
#
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
#
# # use last location to draw a line to the current location
# last_x, last_y, last_z = [], [], []
# plot_cache = [None for _ in range(len(ops))]
#
# # loop each step of the optimization algorithm
# steps = 1000
# plt.show()
# # for iter in range(steps):
# #     for i, op in enumerate(ops):
# #         # run a step of optimization and collect new x and y variable values
# #         # _, x_val, y_val, z_val = sess.run([op, x_var[i], y_var[i], cost[i]])
# #
# #     #     # move dot to the current value
# #     #     if plot_cache[i]:
# #     #         plot_cache[i].remove()
# #     #     plot_cache[i] = ax.scatter(x_val, y_val, z_val, s=3, depthshade=True, label=ops_param[i, 0], color=ops_param[i, 2])
# #     #
# #     #     # draw a line from the previous value
# #     #     if iter == 0:
# #     #         last_z.append(z_val)
# #     #         last_x.append(x_i)
# #     #         last_y.append(y_i)
# #     #     ax.plot([last_x[i], x_val], [last_y[i], y_val], [last_z[i], z_val], linewidth=0.5, color=ops_param[i, 2])
# #     #     last_x[i] = x_val
# #     #     last_y[i] = y_val
# #     #     last_z[i] = z_val
# #     #
# #     # if iter == 0:
# #     #     legend = np.vstack((ops_param[:, 0], ops_param[:, 1])).transpose()
# #     #     plt.legend(plot_cache, legend)
# #     plt.show()
# #
# #     # plt.savefig('figures/' + str(iter) + '.png')
# #     # print('iteration: {}'.format(iter))
# #     #
# #     # plt.pause(0.0001)
#
# print("done")


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


main()
