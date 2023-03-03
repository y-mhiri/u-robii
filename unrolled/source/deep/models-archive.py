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
        self.robust = robust

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
        

        r = 2*nvis # depth of hidden layer 
        self.robust_layer = nn.Sequential(
                    nn.Linear(nvis, r),
                    nn.ReLU(),
                    nn.Linear(r,r),
                    nn.ReLU(),
                    nn.Linear(r, nvis),
                    nn.Sigmoid()
                    )   
        
        if not robust:
            for param in self.robust_layer.parameters():
                param.requires_grad = False

        Wreal = torch.tensor(H.real)
        Wimag = torch.tensor(H.imag)
        self.W.weight = torch.nn.Parameter(torch.complex(Wreal, Wimag))
        self.W.bias =  torch.nn.Parameter(torch.zeros_like(self.W.bias))


    def expectation_step(self, a):
        return self.robust_layer(a)



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
        if self.robust == True:
            tau = self.expectation_step((torch.abs(y- zk).real**2).to(torch.float))
        else: 
            tau = torch.ones((self.nvis))


        tau = torch.squeeze(tau)

        ## M Step
        zk = torch.mul(zk , tau)

        x = xk - self.Wt(zk)/self.nvis
        # x = x + self.Wt(y)/self.nvis
        x = x + self.Wt(torch.mul(y, tau))/self.nvis
        x = torch.sgn(x) * self.softthresh(torch.abs(x).real - self.alpha*torch.max(torch.abs(x)))


        return x

    def forward(self, y):

        x0 = self.Wt(y)
        xk = x0
        for k in range(self.depth):
           
           xk = self.one_iteration(xk, y)

        return torch.abs(xk).real





class UnrolledCNN(nn.Module):
    def __init__(self, depth, npixel, uvw, freq, robust=True, alpha=None):
        super().__init__()
        self.depth = depth
        self.npixel = npixel
        self.robust = robust

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


        
        self.robust_layer_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding='same'),
            nn.AdaptiveMaxPool1d(output_size=16),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.ReLU(),
            )  
         
        self.robust_layer_lin = nn.Sequential(    
                    nn.Linear(16, nvis),
                    nn.Sigmoid()
            )


        Wreal = torch.tensor(H.real)
        Wimag = torch.tensor(H.imag)
        self.W.weight = torch.nn.Parameter(torch.complex(Wreal, Wimag))
        self.W.bias =  torch.nn.Parameter(torch.zeros_like(self.W.bias))


    def expectation_step(self, a):
        x = self.robust_layer_conv(a)
        output = self.robust_layer_lin(x.view(-1, x.size(2), x.size(1)))
        return output




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

            # torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            optimizer.step()

            if batch % 100 == 0:
                loss_val, current = loss.item(), batch * len(x)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

        return loss.item()


    def one_iteration(self, xk, y):

        ## Apply encoder to estimated image
        zk = self.W(xk)
        
        ## Compute robust weight
        if self.robust == True:
            tau = self.expectation_step((torch.abs(zk - y).real**2 / torch.norm(zk - y)**2).to(torch.float))
            # tau = self.robust_layer((torch.abs(zk - y).real**2).to(torch.float))
        else: 
            tau = torch.ones((self.nvis))

        # tau = torch.squeeze(tau)

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




class UnrolledGRU(nn.Module):
    def __init__(self, depth, npixel, uvw, freq, robust=True, alpha=None):
        super().__init__()
        self.depth = depth
        self.npixel = npixel
        self.robust = robust

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


        
        self.update_gate_x = nn.Linear(nvis, nvis)
        self.update_gate_h = nn.Linear(nvis, nvis)
        
        self.reset_gate_x = nn.Linear(nvis, nvis)
        self.reset_gate_h = nn.Linear(nvis, nvis)

        self.new_memory = nn.Linear(nvis, nvis)

        Wreal = torch.tensor(H.real)
        Wimag = torch.tensor(H.imag)
        self.W.weight = torch.nn.Parameter(torch.complex(Wreal, Wimag))
        self.W.bias =  torch.nn.Parameter(torch.zeros_like(self.W.bias))


    def expectation_step(self, xt, wprev):
        zt = nn.Sigmoid()(self.update_gate_x(xt) + self.update_gate_h(wprev))
        rt = nn.Sigmoid()(self.reset_gate_x(xt) + self.reset_gate_h(wprev))

        wnew = nn.Tanh()(xt + torch.mul(rt, self.new_memory(wprev)))
        wnew = torch.mul(wprev, zt) + torch.mul(1-zt, wnew)  

        return wnew




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

            # torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            optimizer.step()

            if batch % 100 == 0:
                loss_val, current = loss.item(), batch * len(x)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

        return loss.item()


    def one_iteration(self, xk, y, tau):

        ## Apply encoder to estimated image
        zk = self.W(xk)
        
        ## Compute robust weight
        if self.robust == True:
            tau = self.expectation_step((torch.abs(zk - y).real**2).to(torch.float), tau)
            # tau = self.expectation_step((torch.abs(zk - y).real**2 / torch.norm(zk - y)**2).to(torch.float), tau)
        else: 
            tau = torch.ones((self.nvis))


        ## M Step
        zk = torch.mul(zk , tau)
        x = xk - self.Wt(zk)/self.nvis
        x = x + self.Wt(torch.mul(y, tau))/self.nvis

        x = torch.sgn(x) * self.softthresh(torch.abs(x).real - self.alpha*torch.max(torch.abs(x)))
        return x, tau

    def forward(self, y):

        x0 = self.Wt(y)
        xk = x0
        
        tau = torch.ones((self.nvis))
        for k in range(self.depth):
           
           xk, tau= self.one_iteration(xk, y, tau)

        return torch.abs(xk).real



class UnrolledLIN(nn.Module):
    def __init__(self, depth, npixel, uvw, freq, robust=True, alpha=None):
        super().__init__()
        self.depth = depth
        self.npixel = npixel
        self.robust = robust

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

        self.estep = nn.Linear(nvis, nvis)

        Wreal = torch.tensor(H.real)
        Wimag = torch.tensor(H.imag)
        self.W.weight = torch.nn.Parameter(torch.complex(Wreal, Wimag))
        self.W.bias =  torch.nn.Parameter(torch.zeros_like(self.W.bias))


    def expectation_step(self, xt, wprev):
        return nn.Sigmoid()(self.estep(xt) + wprev)




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

            # torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            optimizer.step()

            if batch % 100 == 0:
                loss_val, current = loss.item(), batch * len(x)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

        return loss.item()


    def one_iteration(self, xk, y, tau):

        ## Apply encoder to estimated image
        zk = self.W(xk)
        
        ## Compute robust weight
        if self.robust == True:
            tau = self.expectation_step((torch.abs(zk - y).real**2 / torch.norm(zk - y)**2).to(torch.float), tau)
        else: 
            tau = torch.ones((self.nvis))


        ## M Step
        zk = torch.mul(zk , tau)
        x = xk - self.Wt(zk)/self.nvis
        x = x + self.Wt(torch.mul(y, tau))/self.nvis

        x = torch.sgn(x) * self.softthresh(torch.abs(x).real - self.alpha*torch.max(torch.abs(x)))
        return x, tau

    def forward(self, y):

        x0 = self.Wt(y)
        xk = x0
        
        tau = torch.ones((self.nvis))
        for k in range(self.depth):
           
           xk, tau= self.one_iteration(xk, y, tau)

        return torch.abs(xk).real



class UnrolledW(nn.Module):
    def __init__(self, depth, npixel, uvw, freq, robust=True, alpha=None):
        super().__init__()
        self.depth = depth
        self.npixel = npixel
        self.robust = robust

        H = forward_operator(uvw, freq, npixel)
        nvis, npix2 = H.shape
         
        self.nvis = nvis

        self.W = nn.Linear(npix2,  nvis).to(torch.cdouble)
        self.Wt = lambda a : torch.matmul(a, self.W.weight.conj()/npix2)

        if alpha is not None:
            self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True))
        else:
            self.alpha = nn.Parameter(torch.tensor(0.1, requires_grad=True))
            
        self.robust_weights = nn.Parameter(torch.ones((depth, nvis)), requires_grad=True)
        self.softthresh = nn.ReLU()


        Wreal = torch.tensor(H.real)
        Wimag = torch.tensor(H.imag)
        self.W.weight = torch.nn.Parameter(torch.complex(Wreal, Wimag))
        self.W.bias =  torch.nn.Parameter(torch.zeros_like(self.W.bias))


    def expectation_step(self, xt, k):
        return self.robust_weights[k]




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

            # torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            optimizer.step()

            if batch % 100 == 0:
                loss_val, current = loss.item(), batch * len(x)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

        return loss.item()


    def one_iteration(self, xk, y, k):

        ## Apply encoder to estimated image
        zk = self.W(xk)
        
        ## Compute robust weight
        if self.robust == True:
            tau = self.expectation_step((torch.abs(zk - y).real**2 / torch.norm(zk - y)**2).to(torch.float), k)
        else: 
            tau = torch.ones((self.nvis))


        ## M Step
        zk = torch.mul(zk , tau)
        x = xk - self.Wt(zk)/self.nvis
        x = x + self.Wt(torch.mul(y, tau))/self.nvis

        x = torch.sgn(x) * self.softthresh(torch.abs(x).real - self.alpha*torch.max(torch.abs(x)))
        return x, tau

    def forward(self, y):

        x0 = self.Wt(y)
        xk = x0
        
        tau = torch.ones((self.nvis))
        for k in range(self.depth):
           
           xk, tau= self.one_iteration(xk, y, k)

        return torch.abs(xk).real



def eval_net(vis, path, *args, **kwargs):

    threshold = kwargs["alpha"]
    net = UnrolledCNN(*args, **kwargs)
    # net.robust = False
    if path is not None:
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint["model_state_dict"])
        if threshold is not None:
            net.set_threshold(threshold)

    pred = net(ToTensor()(vis.reshape(1,-1))).detach().numpy()

    return pred.reshape(net.npixel, net.npixel)



# class UnrolledFixed(Unrolled):
#     def __init__(self, depth, npixel, uvw, freq, robust=True, alpha=None):
#         super().__init__(depth, npixel, uvw, freq, robust, alpha)

#         self.robust_weights = nn.Parameter(torch.ones((depth, self.nvis)), requires_grad=True)
#         self.trainable_robust = [self.robust_weights]

        
#     def expectation_step(self, k):
#         return self.robust_weights[k]
    

#     def one_iteration(self, xk, y, tau):

#         ## Apply encoder to estimated image
#         zk = self.W(xk)

#         ## Compute robust weight
#         tau = self.expectation_step((torch.abs(y- zk).real**2).to(torch.float), tau)

#         ## M Step
#         zk = torch.mul(zk , tau)
#         x = xk - self.Wt(zk)/self.nvis
#         x = x + self.Wt(torch.mul(y, tau))/self.nvis
#         x = torch.sgn(x) * self.softthresh(torch.abs(x).real - self.alpha*torch.max(torch.abs(x)))


#         return x, tau


#     def forward(self, y):

#         x0 = self.Wt(y)
#         xk = x0

#         tau = torch.ones(self.nvis)
#         for k in range(self.depth):
           
#            xk, tau = self.one_iteration(xk, y, tau)

#         return torch.abs(xk).real



# class UnrolledLIN(nn.Module):
#     def __init__(self, depth, npixel, uvw, freq, robust=True, alpha=None):
#         super().__init__()
#         self.depth = depth
#         self.npixel = npixel
#         self.robust = robust

#         H = forward_operator(uvw, freq, npixel)
#         nvis, npix2 = H.shape
         
#         self.nvis = nvis

#         self.W = nn.Linear(npix2,  nvis).to(torch.cdouble)
#         self.Wt = lambda a : torch.matmul(a, self.W.weight.conj()/npix2)

#         if alpha is not None:
#             self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True))
#         else:
#             self.alpha = nn.Parameter(torch.tensor(0.1, requires_grad=True))
            
#         self.softthresh = nn.ReLU()

#         self.estep = nn.Linear(nvis, nvis)

#         Wreal = torch.tensor(H.real)
#         Wimag = torch.tensor(H.imag)
#         self.W.weight = torch.nn.Parameter(torch.complex(Wreal, Wimag))
#         self.W.bias =  torch.nn.Parameter(torch.zeros_like(self.W.bias))


#     def expectation_step(self, xt, wprev):
#         return nn.ReLU()(self.estep(xt) + wprev)




#     def set_threshold(self, alpha):
#         self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=False))
#         return True

#     def compute_loss(self, dataloader, loss_fn, device):

#         size = len(dataloader.dataset)

#         for batch, (y, x) in enumerate(dataloader):
#             y, x = y.to(device), x.to(device)

#             # Compute prediction error
#             pred = self(y)
#             # loss = loglike_fn(pred, y)
#             loss = loss_fn(pred, x)

#             if batch % 100 == 0:
#                 loss_val, current = loss.item(), batch * len(x)
#                 print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

#         return loss.item()      


#     def train_supervised(self, dataloader, loss_fn, optimizer, device):

#         size = len(dataloader.dataset)
#         self.train()
#         for batch, (y, x) in enumerate(dataloader):
#             y, x = y.to(device), x.to(device)

#             # Compute prediction error
#             pred = self(y)
#             loss = loss_fn(pred, x)

#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()

#             # torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
#             optimizer.step()

#             if batch % 100 == 0:
#                 loss_val, current = loss.item(), batch * len(x)
#                 print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

#         return loss.item()


#     def one_iteration(self, xk, y, tau):

#         ## Apply encoder to estimated image
#         zk = self.W(xk)
        
#         ## Compute robust weight
#         if self.robust == True:
#             tau = self.expectation_step((torch.abs(zk - y).real**2 / torch.norm(zk - y)**2).to(torch.float), tau)
#         else: 
#             tau = torch.ones((self.nvis))


#         ## M Step
#         zk = torch.mul(zk , tau)
#         x = xk - self.Wt(zk)/self.nvis
#         x = x + self.Wt(torch.mul(y, tau))/self.nvis

#         x = torch.sgn(x) * self.softthresh(torch.abs(x).real - self.alpha*torch.max(torch.abs(x)))
#         return x, tau

#     def forward(self, y):

#         x0 = self.Wt(y)
#         xk = x0
        
#         tau = torch.ones((self.nvis))
#         for k in range(self.depth):
           
#            xk, tau= self.one_iteration(xk, y, tau)

#         return torch.abs(xk).real



# class UnrolledW(nn.Module):
#     def __init__(self, depth, npixel, uvw, freq, robust=True, alpha=None):
#         super().__init__()
#         self.depth = depth
#         self.npixel = npixel
#         self.robust = robust

#         H = forward_operator(uvw, freq, npixel)
#         nvis, npix2 = H.shape
         
#         self.nvis = nvis

#         self.W = nn.Linear(npix2,  nvis).to(torch.cdouble)
#         self.Wt = lambda a : torch.matmul(a, self.W.weight.conj()/npix2)

#         if alpha is not None:
#             self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True))
#         else:
#             self.alpha = nn.Parameter(torch.tensor(0.1, requires_grad=True))
            
#         self.robust_weights = nn.Parameter(torch.ones((depth, nvis)), requires_grad=True)
#         self.softthresh = nn.ReLU()


#         Wreal = torch.tensor(H.real)
#         Wimag = torch.tensor(H.imag)
#         self.W.weight = torch.nn.Parameter(torch.complex(Wreal, Wimag))
#         self.W.bias =  torch.nn.Parameter(torch.zeros_like(self.W.bias))


#     def expectation_step(self, xt, k):
#         return self.robust_weights[k]




#     def set_threshold(self, alpha):
#         self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=False))
#         return True

#     def compute_loss(self, dataloader, loss_fn, device):

#         size = len(dataloader.dataset)

#         for batch, (y, x) in enumerate(dataloader):
#             y, x = y.to(device), x.to(device)

#             # Compute prediction error
#             pred = self(y)
#             # loss = loglike_fn(pred, y)
#             loss = loss_fn(pred, x)

#             if batch % 100 == 0:
#                 loss_val, current = loss.item(), batch * len(x)
#                 print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

#         return loss.item()      


#     def train_supervised(self, dataloader, loss_fn, optimizer, device):

#         size = len(dataloader.dataset)
#         self.train()
#         for batch, (y, x) in enumerate(dataloader):
#             y, x = y.to(device), x.to(device)

#             # Compute prediction error
#             pred = self(y)
#             loss = loss_fn(pred, x)

#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()

#             # torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
#             optimizer.step()

#             if batch % 100 == 0:
#                 loss_val, current = loss.item(), batch * len(x)
#                 print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

#         return loss.item()


#     def one_iteration(self, xk, y, k):

#         ## Apply encoder to estimated image
#         zk = self.W(xk)
        
#         ## Compute robust weight
#         if self.robust == True:
#             tau = self.expectation_step((torch.abs(zk - y).real**2 / torch.norm(zk - y)**2).to(torch.float), k)
#         else: 
#             tau = torch.ones((self.nvis))


#         ## M Step
#         zk = torch.mul(zk , tau)
#         x = xk - self.Wt(zk)/self.nvis
#         x = x + self.Wt(torch.mul(y, tau))/self.nvis

#         x = torch.sgn(x) * self.softthresh(torch.abs(x).real - self.alpha*torch.max(torch.abs(x)))
#         return x, tau

#     def forward(self, y):

#         x0 = self.Wt(y)
#         xk = x0
        
#         tau = torch.ones((self.nvis))
#         for k in range(self.depth):
           
#            xk, tau= self.one_iteration(xk, y, k)

#         return torch.abs(xk).real





# class UnrolledCNN(Unrolled):
#     def __init__(self, depth, npixel, uvw, freq, robust=True, alpha=None):
#         super().__init__(depth, npixel, uvw, freq, robust, alpha)

#         self.robust_layer_conv = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=5, padding='same'),
#             nn.AdaptiveMaxPool1d(output_size=16),
#             nn.ReLU(),
#             nn.Conv1d(32, 16, kernel_size=5),
#             nn.AdaptiveAvgPool1d(output_size=1),
#             nn.ReLU(),
#             )  
         
#         self.robust_layer_lin = nn.Sequential(    
#                     nn.Linear(16, self.nvis),
#                     nn.Sigmoid()
#             )
        
#         self.trainable_robust = []
#         for param in self.robust_layer_conv.parameters():
#             self.trainable_robust.append(param)
#         for param in self.robust_layer_lin.parameters():
#             self.trainable_robust.append(param)


#     def expectation_step(self, a):
#         x = self.robust_layer_conv(a)
#         output = self.robust_layer_lin(x.view(-1, x.size(2), x.size(1)))
#         return output



# class UnrolledGRU(Unrolled):
#     def __init__(self, depth, npixel, uvw, freq, robust=True, alpha=None):
#         super().__init__(depth, npixel, uvw, freq, robust, alpha)

#         self.update_gate_x = nn.Linear(self.nvis, self.nvis)
#         self.update_gate_h = nn.Linear(self.nvis, self.nvis)
        
#         self.reset_gate_x = nn.Linear(self.nvis, self.nvis)
#         self.reset_gate_h = nn.Linear(self.nvis, self.nvis)

#         self.new_memory = nn.Linear(self.nvis, self.nvis)

#         self.robust = nn.Linear(self.nvis, self.nvis)


#         self.trainable_robust = []
#         for param in self.update_gate_x.parameters():
#             self.trainable_robust.append(param)
#         for param in self.update_gate_h.parameters():
#             self.trainable_robust.append(param)
#         for param in self.reset_gate_x.parameters():
#             self.trainable_robust.append(param)
#         for param in self.reset_gate_h.parameters():
#             self.trainable_robust.append(param)
#         for param in self.new_memory.parameters():
#             self.trainable_robust.append(param)

        
#     def expectation_step(self, xt, wprev):
#         zt = nn.Sigmoid()(self.update_gate_x(xt) + self.update_gate_h(wprev))
#         rt = nn.Sigmoid()(self.reset_gate_x(xt) + self.reset_gate_h(wprev))

#         wnew = nn.ReLU()(self.robust(1/xt)) + torch.mul(rt, self.new_memory(wprev))
#         wnew = torch.mul(wprev, zt) + torch.mul(1-zt, wnew)  

#         return wnew

#     def one_iteration(self, xk, y, tau):

#         ## Apply encoder to estimated image
#         zk = self.W(xk)

#         ## Compute robust weight
#         tau = self.expectation_step((torch.abs(y- zk).real**2).to(torch.float), tau)

#         ## M Step
#         zk = torch.mul(zk , tau)
#         x = xk - self.Wt(zk)/self.nvis
#         x = x + self.Wt(torch.mul(y, tau))/self.nvis
#         x = torch.sgn(x) * self.softthresh(torch.abs(x).real - self.alpha*torch.max(torch.abs(x)))


#         return x, tau

#     def forward(self, y):

#         x0 = self.Wt(y)
#         xk = x0

#         tau = torch.ones(self.nvis)
#         for k in range(self.depth):
           
#            xk, tau = self.one_iteration(xk, y, tau)

#         return torch.abs(xk).real
