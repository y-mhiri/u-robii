import numpy as np
from ducc0.wgridder import ms2dirty, dirty2ms
from copy import deepcopy

from ..deep.models import RobustLayer
from torchvision.transforms import ToTensor


def robust_ml_imager(vis, uvw, freq, cellsize, niter, 
                model_image=None, npix_x=32, npix_y=32, 
                nu=10, alpha=0.01, gaussian=False, gamma=1,
                miter=1, debug=False):

    """
    The product by the model matrix or its adjoint is done with dirty2ms and ms2dirty


    """
    vis = vis.reshape(-1)
    nvis  =  len(vis)

    if model_image == None:
        model_image = ms2dirty(  
                                uvw = uvw,
                                freq = freq,
                                ms = vis.reshape(-1,1),
                                npix_x = npix_x,
                                npix_y = npix_y,
                                pixsize_x = cellsize,
                                pixsize_y = cellsize,
                                epsilon=1.0e-7 
                        )/nvis

    model_image_k = deepcopy(model_image) 
    npix_x, npix_y = model_image.shape

    expected_tau = np.ones(nvis)

    for it in range(niter):
        
        model_vis = dirty2ms(  
                            uvw = uvw,
                            freq = freq,
                            dirty = model_image_k,
                            pixsize_x = cellsize,
                            pixsize_y = cellsize,
                            epsilon=1.0e-7 
                    )

        residual = (vis.reshape(-1) - model_vis.reshape(-1))
        sigma2 = (1/nvis) * np.linalg.norm(np.diag(np.sqrt(expected_tau)) @ residual)**2
        expected_tau = (nu + 1)/(nu + (1/sigma2) * np.linalg.norm(residual.reshape(-1,1), axis=1)**2) 
        if gaussian:
            expected_tau = np.ones(nvis) 

        
        for mit in range(miter):
            model_vis = dirty2ms(  
                                uvw = uvw,
                                freq = freq,
                                dirty = model_image_k,
                                pixsize_x = cellsize,
                                pixsize_y = cellsize,
                                epsilon=1.0e-7 
                        )

            residual = (vis.reshape(-1) - model_vis.reshape(-1))
            residual = np.diag(expected_tau) @ residual.reshape(-1)

            residual_image = ms2dirty(  
                                uvw = uvw,
                                freq = freq,
                                ms = residual.reshape(-1,1),
                                npix_x = npix_x,
                                npix_y = npix_y,
                                pixsize_x = cellsize,
                                pixsize_y = cellsize,
                                epsilon=1.0e-7 
                        )/nvis
        
            model_image_k = model_image_k + gamma*residual_image
            model_image_k = np.sign(model_image_k) * np.max([np.abs(model_image_k)- alpha*np.max(np.abs(model_image_k)), np.zeros(model_image_k.shape)], axis=0)
            model_image_k = np.abs(model_image_k)

            
    return np.abs(model_image_k)



    
def unrolledImager(vis, model_path, freq, uvw, npix_x, npix_y, cellsize, niter):

    """
    Imager Algorithm that uses as Expectation Step a recurent neural network 
    to compute the weights applied to the residual
    
    """

    nvis = len(uvw)
    net = RobustLayer(nvis)
    
    checkpoint = torch.load(model_path)
    
    checkpoint["model_state_dict"].pop('W.weight')
    checkpoint["model_state_dict"].pop('W.bias')
    checkpoint["model_state_dict"].pop('alpha')

    net.load_state_dict(checkpoint["model_state_dict"])


    model_image = ms2dirty(  
                            uvw = uvw,
                            freq = freq,
                            ms = vis.reshape(-1,1),
                            npix_x = npix_x,
                            npix_y = npix_y,
                            pixsize_x = cellsize,
                            pixsize_y = cellsize,
                            epsilon=1.0e-7 
                        )/nvis/(npix_x*npix_y)

    model_image_k = deepcopy(model_image) 
    expected_weights = np.ones(nvis) 

    for it in range(niter):

        model_vis = dirty2ms(  
                        uvw = uvw,
                        freq = freq,
                        dirty = model_image_k,
                        pixsize_x = cellsize,
                        pixsize_y = cellsize,
                        epsilon=1.0e-7 
                )

        residual = (vis.reshape(-1) - model_vis.reshape(-1)).reshape(1,-1)

        expected_weights = net(ToTensor()(np.linalg.norm(residual, axis=0).reshape(1,-1).astype(np.float32)**2),
                                 ToTensor()(expected_weights.reshape(1,-1).astype(np.float32))) 
        expected_weights = expected_weights.detach().numpy().squeeze()
        residual = np.diag(expected_weights) @ residual.reshape(-1)
        residual_image = ms2dirty(  
                                uvw = uvw,
                                freq = freq,
                                ms = residual.reshape(-1,1),
                                npix_x = npix_x,
                                npix_y = npix_y,
                                pixsize_x = cellsize,
                                pixsize_y = cellsize,
                                epsilon=1.0e-7 
                        )/nvis/(npix_x*npix_y)
        
        model_image_k = model_image_k + residual_image
        model_image_k = np.sign(model_image_k) * np.max([np.abs(model_image_k)- 0.1*np.max(np.abs(model_image_k)), np.zeros(model_image_k.shape)], axis=0)
        model_image_k = np.abs(model_image_k)

            
    return np.abs(model_image_k)
