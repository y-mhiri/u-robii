import numpy as np
from ..astro.astro import generate_directions
from scipy.constants import speed_of_light

import torch
from torchvision.transforms import ToTensor

from torch import nn

"""
Pytorch Models and operators


"""


def forward_operator(uvw, freq, npixel, cellsize=None):
    """
    Used to initialize the model matrix

    """
    wl = speed_of_light / freq
    if cellsize is None:
        cellsize = np.min(wl)/(2*np.max(uvw))

    # cellsize = np.rad2deg(cellsize)
    # fov = cellsize*npixel
    lmn = generate_directions(npixel, cellsize).reshape(3,-1)
    uvw = uvw.reshape(-1,3)

    return np.exp(-1j*(2*np.pi/np.min(wl))* uvw @ lmn)


class Unrolled(nn.Module):
    def __init__(self, depth, npixel, uvw, freq, robust=True, alpha=None):
        super().__init__()
        self.depth = depth
        self.npixel = npixel

        H = forward_operator(uvw, freq, npixel)
        nvis, npix2 = H.shape
         
        self.nvis = nvis

        self.W = nn.Linear(npix2,  nvis).to(torch.cdouble)
        self.Wt = lambda a : torch.matmul(a, self.W.weight.conj()/npix2)

        if alpha is not None:
            self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True))
        else:
            self.alpha = nn.Parameter(torch.tensor(0.1, requires_grad=True))
            
        self.softthresh = nn.ReLU()
          
        Wreal = torch.tensor(H.real)
        Wimag = torch.tensor(H.imag)
        self.W.weight = torch.nn.Parameter(torch.complex(Wreal, Wimag))
        self.W.bias =  torch.nn.Parameter(torch.zeros_like(self.W.bias))

        self.trainable_model = []
        self.trainable_model.append(self.W.weight)
        self.trainable_model.append(self.alpha)


    def expectation_step(self, a):
        return torch.ones((self.nvis))

    def set_threshold(self, alpha):
        self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=False))
        return True

    def compute_loss(self, dataloader, loss_fn, device):

        size = len(dataloader.dataset)

        for batch, (y, x) in enumerate(dataloader):
            y, x = y.to(device), x.to(device)

            # Compute prediction error
            pred = self(y)
            # loss = loglike_fn(pred, y)
            loss = loss_fn(pred, x)

            if batch % 100 == 0:
                loss_val, current = loss.item(), batch * len(x)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

        return loss.item()      


    def train_supervised(self, dataloader, loss_fn, optimizer, device):

        size = len(dataloader.dataset)
        self.train()
        for batch, (y, x) in enumerate(dataloader):
            y, x = y.to(device), x.to(device)

            # Compute prediction error
            pred = self(y)
            loss = loss_fn(pred, x)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss_val, current = loss.item(), batch * len(x)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

        return loss.item()


    def one_iteration(self, xk, y):

        ## Apply encoder to estimated image
        zk = self.W(xk)

        ## Compute robust weight
        tau = self.expectation_step((torch.abs(y- zk).real**2).to(torch.float))

        ## M Step
        zk = torch.mul(zk , tau)
        x = xk - self.Wt(zk)/self.nvis
        x = x + self.Wt(torch.mul(y, tau))/self.nvis
        x = torch.sgn(x) * self.softthresh(torch.abs(x).real - self.alpha*torch.max(torch.abs(x)))


        return x

    def forward(self, y):

        x0 = self.Wt(y)
        xk = x0
        for k in range(self.depth):
           
           xk = self.one_iteration(xk, y)

        return torch.abs(xk).real

class RobustLayer(nn.Module):
    def __init__(self, nvis):
        super().__init__()

        self.estep = nn.Linear(self.nvis, self.nvis)
        self.update_layer = nn.Linear(self.nvis, self.nvis)
        self.memory_layer = nn.Linear(self.nvis, self.nvis)


    def forward(self, residual, wprev):
        wtemp = nn.Sigmoid()(self.estep(residual))
        wnew = nn.ReLU()(self.update_layer(wtemp) + self.memory_layer(wprev))
        return wnew
        

class UnrolledRobust(Unrolled):
    def __init__(self, depth, npixel, uvw, freq, robust=True, alpha=None):
        super().__init__(depth, npixel, uvw, freq, robust, alpha)

        self.estep = nn.Linear(self.nvis, self.nvis)
        self.update_layer = nn.Linear(self.nvis, self.nvis)
        self.memory_layer = nn.Linear(self.nvis, self.nvis)
        self.trainable_robust = []
        for param in self.estep.parameters():
            self.trainable_robust.append(param)
        for param in self.update_layer.parameters():
            self.trainable_robust.append(param)
        for param in self.memory_layer.parameters():
            self.trainable_robust.append(param)
        
    def expectation_step(self, xt, wprev):
        wtemp = nn.Sigmoid()(self.estep(xt))
        wnew = nn.ReLU()(self.update_layer(wtemp) + self.memory_layer(wprev))
        return wnew
        # return nn.ReLU()(self.estep(1/xt))
    

    def one_iteration(self, xk, y, tau):

        ## Apply encoder to estimated image
        zk = self.W(xk)

        ## Compute robust weight
        tau = self.expectation_step((torch.abs(y- zk).real**2).to(torch.float), tau)

        ## M Step
        zk = torch.mul(zk , tau)
        x = xk - self.Wt(zk)/self.nvis
        x = x + self.Wt(torch.mul(y, tau))/self.nvis
        x = torch.sgn(x) * self.softthresh(torch.abs(x).real - self.alpha*torch.max(torch.abs(x)))


        return x, tau


    def forward(self, y):

        x0 = self.Wt(y)
        xk = x0

        tau = torch.ones(self.nvis)
        for k in range(self.depth):
           
           xk, tau = self.one_iteration(xk, y, tau)

        return torch.abs(xk).real
    


def eval_net(vis, path, *args, **kwargs):

    threshold = kwargs["alpha"]
    net = UnrolledRobust(*args, **kwargs)
    # net.robust = False
    if path is not None:
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint["model_state_dict"])
        if threshold is not None:
            net.set_threshold(threshold)

    pred = net(ToTensor()(vis.reshape(1,-1))).detach().numpy()

    return pred.reshape(net.npixel, net.npixel)




