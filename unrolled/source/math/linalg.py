import numpy as np
from numpy.linalg import norm

from skimage import data, img_as_float
from skimage.metrics import structural_similarity, peak_signal_noise_ratio




def H(_M):
    return _M.conjugate().transpose()
def vec(M):
    return M.reshape((-1,1), order='F')
def unvec(M, shape):
    return M.reshape(shape)
    
def kron_fast(A,B):
    a = A[:,np.newaxis,:,np.newaxis]
    a = A[:,np.newaxis,:,np.newaxis]*B[np.newaxis,:,np.newaxis,:]
    a.shape = (A.shape[0]*B.shape[0],A.shape[1]*B.shape[1])
    return a
def inv_diagonal(_M, _bloc_size):

    """ 
        Fast implementation of the inverse for block diagonal matrix
    """
    _shape = _M.shape
    if len(_shape) != 2:
        raise ValueError("Not a square Matrix")
    if _shape[0] != _shape[1]:
        raise ValueError("Not a square Matrix")
    if _shape[0] % _bloc_size != 0:
        raise ValueError("Invalid Bloc Size")

    _n_bloc = _shape[0] // _bloc_size

    _M_inv = np.zeros(_M.shape).astype(np.complex)
    for k in range(_n_bloc):
        _M_inv[_bloc_size * k:_bloc_size * (k + 1), _bloc_size * k:_bloc_size * (k + 1)] = \
            np.linalg.pinv(_M[_bloc_size * k:_bloc_size * (k + 1), _bloc_size * k:_bloc_size * (k + 1)])
    return _M_inv



def fourier_matrix(N, F):
    n = np.linspace(0, N-1, N).reshape((1, -1))
    f = np.linspace(-F//2, F//2-1, F).reshape((-1, 1))
    
    return np.exp(-1j*2*np.pi*n*f/N)


def compute_mse(*args):

    MSE = np.zeros(len(args)).astype(np.complex)
    for i in range(len(args)):
        _x = args[i][1].reshape((-1, 1))
        _x_est = args[i][0].reshape((-1, 1))

        MSE[i] = np.linalg.norm(_x - _x_est)**2 / (np.linalg.norm(_x))**2
        #MSE[i] = np.sqrt(np.linalg.norm(_x - _x_est)**2 / np.product([ k for k in _x.shape]))

    return MSE

def fourier_matrix(N, flat=False):

    F2D = np.zeros((N,N)).astype(complex)
    n = np.linspace(0, N-1, N).reshape((1, -1))
    f = np.linspace(-N//2, N//2-1, N).reshape((-1, 1))
    
    F2D = np.exp(-1j*2*np.pi*(f@n)/N)

    if flat:
        l = np.hstack([k*np.ones(N) for k in range(N)]).reshape(1,-1)
        u = np.hstack([(k-N//2)*np.ones(N) for k in range(N)]).reshape(-1,1)
        
        m = np.hstack([np.arange(N) for k in range(N)]).reshape(1,-1)
        v = np.hstack([np.arange(N)- N//2 for k in range(N)]).reshape(-1,1)

        F1D = np.exp(-1j*2*np.pi*(u@l + v@m)/N)
        return F1D

    return F2D 



def psnr(estimated_image, true_image):
    estimated_image = estimated_image/(np.max(estimated_image) + 1e-6)
    true_image = true_image/(np.max(true_image) + 1e-6)

    err = norm(estimated_image - true_image)**2
    return 10 * np.log10(1 / err)

def ssim(estimated_image, true_image, *args, **kwargs):
    estimated_image = estimated_image/(np.max(estimated_image) + 1e-6)
    true_image = true_image/(np.max(true_image) + 1e-6)

    return structural_similarity(estimated_image, true_image, data_range=1, *args, **kwargs)

def nmse(estimated_image, true_image, reduce='norm'):
    if reduce == 'norm':
        return norm(estimated_image - true_image)**2 /  norm(true_image)**2
    if reduce == 'dim':
        return norm(estimated_image - true_image)**2 /  len(estimated_image.flatten())

def snr(estimated_image, true_image):
    return 10*np.log10(1/nmse(estimated_image, true_image))

def normalized_cross_correlation(X, Y):

    N = len(X.reshape(-1))

    sigmaX, sigmaY = np.sqrt(np.var(X)), np.sqrt(np.var(Y))
    muX, muY = np.mean(X), np.mean(Y)

    Xvec, Yvec = X.reshape(-1), Y.reshape(-1)

    return np.sum( [(Xi - muX)*(Yi - muY) for Xi,Yi in zip(Xvec,Yvec)])/(N*sigmaX*sigmaY)
    