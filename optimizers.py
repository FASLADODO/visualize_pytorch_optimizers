import torch

import losses


class Optimizer:
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
        loss = losses.compute_loss(self.w1, self.w2)
        loss.backward()
        self.opt.step()

        return loss

    def update(self):
        self.w1_list += [float(self.w1)]
        self.w2_list += [float(self.w2)]
        self.loss_list += [float(losses.compute_loss(self.w1, self.w2))]

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
