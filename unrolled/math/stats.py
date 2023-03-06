import numpy as np
from scipy.stats import t, norm, invgamma
from scipy.special import gamma


def compute_log_likelihood(_data, distribution = "gaussian", **kwargs):

    if distribution == "gaussian":
        _mean, _Cov = kwargs["mean"] , kwargs["Cov"]
        sign, logdet = np.linalg.slogdet(_Cov)
        L =  logdet + (_data - _mean).T.conj() @ np.linalg.solve(_Cov, _data - _mean)

    if distribution == "student-t":
        _mean, _Cov, _nu = kwargs["mean"] , kwargs["Cov"], kwargs["nu"]
        p = len(_data)
        _, logdet = np.linalg.slogdet(_Cov)
        cste = gamma((_nu+p)/2)/((_nu*np.pi)**(p/2) * gamma(_nu/2)) 
        
        L = 0.5*logdet + np.log(cste * (1 + (_data - _mean).T.conj() @ np.linalg.solve(_Cov, _data - _mean)/_nu)**(-(_nu + p)/2))

    
    return L.reshape((-1))


def complex_normal(mean, Cov, diag=True, rng=np.random.RandomState(0)): 

        n_samples = len(mean)
        # Check if covariance matrix is symetric
        if not np.allclose(Cov,Cov.T.conjugate(), atol=1e-08):
           raise ValueError("Covariance matrix must be symetric.")

        SIGMA = np.zeros((n_samples*2, n_samples*2))

        SIGMA[0: n_samples, 0:n_samples] = np.real(Cov)
        SIGMA[n_samples: 2*n_samples, n_samples:2*n_samples] = np.real(Cov)
        
        SIGMA[0: n_samples, n_samples:2*n_samples] = -np.imag(Cov)
        SIGMA[n_samples: 2*n_samples, 0: n_samples] = np.imag(Cov)

        SIGMA = (1/2)*SIGMA

        if not diag:
            S,D,V = np.linalg.svd(SIGMA)

            # if not np.allclose(S,V.T.conjugate(), rtol=1):
            #     print(S - V.T.conjugate())
            #     raise ValueError("SVD - Covariance matrix is not symetric")

            S = np.dot(S, np.diag(np.sqrt(D)))

        else:
            S = np.sqrt(SIGMA)

        MU = np.zeros(2*n_samples)
        MU[0:n_samples] = np.real(mean).reshape(-1)
        MU[n_samples:2*n_samples] = np.imag(mean).reshape(-1)
        
        _y = np.dot(S , rng.normal(0, 1, 2*n_samples)) + MU

        return (_y[0:n_samples] + 1j*_y[n_samples::]).reshape((-1,1))
        
        
def fast_complex_normal(mean, Cov, diag=True, rng=np.random.RandomState(0)): 
    """
        Here for compatibility reasons
    """

    return complex_normal(mean, Cov, diag=diag, rng=rng)
    


def gaussian_pdf(x, loc, scale):

    if x.dtype == 'float':
        return norm.pdf(x, loc=loc, scale=scale)
    elif x.dtype == 'complex':
        return ((1/ (np.pi*scale**2))*np.exp(-(x-loc)*(x-loc).conj() / scale**2)).real
    
    else:
        raise ValueError("Unknown type")


def student_pdf(x, df, loc, scale):

    if x.dtype == 'float':
        return t.pdf(x, df=df, loc=loc, scale=scale)
    elif x.dtype == 'complex':
        return (gamma(df/2+1)/gamma(df/2) ) * ((2/(np.pi*df*scale**2))*(1 + 2*(x-loc)*(x-loc).conj()/(df*scale**2))**(-df/2 -1)).real
        # return (gamma(df+1)/gamma(df) ) * ((1/(np.pi*df*scale**2))*(1 + (x-loc)*(x-loc).conj()/(df*scale**2))**(-df-1)).real
    
    else:
        raise ValueError("Unknown type")