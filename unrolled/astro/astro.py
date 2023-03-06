import os
import shutil
import numpy as np
import zarr
import numcodecs
import datetime
import click

from ..math.linalg import vec
from ..math.stats import complex_normal
from scipy.stats import invgamma, gamma, invgauss

from .ms import MS
from ducc0.wgridder import dirty2ms, ms2dirty
from scipy.constants import speed_of_light
from casacore.tables import table 

from copy import deepcopy
from PIL import Image
from omegaconf import OmegaConf

from .. import ROOT_DIR




def get_path():
    print(ROOT_DIR)

def generate_directions(npix, cellsize):


    """
    generate direction cosines vectors (l,m,n)
    for an image of size (npix, npix) and a cellsize defined in radians
    Input :
    - (int) npix : image pixel size
    - (float) cellsize : size of a pixel in radians

    Returns : (ndarray) lmn 
    """

    k = np.arange(0, npix)
    l_grid = (-npix/2 + k)*cellsize
    m_grid = (-npix/2 + k)*cellsize


    LGRID, MGRID = np.meshgrid(l_grid, m_grid)
    NGRID = np.sqrt(1 - LGRID**2 - MGRID**2)


    l = vec(LGRID)
    m = vec(MGRID)
    n = vec(NGRID)

    lmn = np.array([l,m,n]).reshape(3,-1)

    # lmn = np.array([l,m,n]).T

    return lmn

def gaussian_source(center, P, scale, npixel):
    X,Y = np.meshgrid(np.linspace(-npixel//2, npixel//2, npixel),np.linspace(-npixel//2, npixel//2, npixel))
    kernel = P * np.exp(-((X-center[0]*np.ones_like(X))**2 + (Y-center[1]*np.ones_like(Y))**2)/scale**2)
    return kernel

def generate_sky_model(sources, npixel, rng=np.random.default_rng()):


    skymodel = np.zeros((npixel, npixel))
    for s in sources:
        center = s["position"]
        P = s["power"]
        scale = s["scale"]
        skymodel += gaussian_source(center, P, scale, npixel)

    return skymodel

def random_skymodel(nsources, npixel, pmin=0, pmax=1, sig_min=1, sig_max=10, fixed=False, positions=None, rng=np.random.default_rng()):

    
    if fixed:
        assert nsources == len(positions)
        source_position = np.array(positions)
    else:
        source_position = rng.uniform(-npixel//2, npixel//2,(nsources, 2))

    source_power = rng.uniform(pmin , pmax, nsources)
    source_scale = rng.uniform(sig_min , sig_max, nsources)

    sources = [{"position" : pos,
                "power" : power,
                "scale" : scale} for pos, power, scale
                in zip(source_position, source_power, source_scale)]

    return generate_sky_model(sources, npixel, rng)



@click.command()
@click.argument('configfile', type=click.Path(exists=True))
@click.option('--seed', default=-1, help='Seed value for RandomState')
def generate_dataset(configfile, seed):

    if seed == -1:
        rng = np.random.default_rng()
    else:
        rng = rng = np.random.default_rng(seed)
    metadata = locals()

    conf = OmegaConf.load(configfile)

    name = conf.info.name
    # dirname = os.path.dirname(os.path.abspath(configfile))
    path = f'{name}.zip'
    msname = conf.info.observation_profile
    nimage = conf.info.nimage
    npixel = conf.info.npixel
    coplanar = conf.info.coplanar

    from_sky_model = conf.sky_model.from_sky_model
    sky_model_path = conf.sky_model.path

    fixed = conf.source.positions.fixed
    positions = conf.source.positions.coordinates

    nsource_min = conf.source.nsource.min
    nsource_max = conf.source.nsource.max
    pmin = conf.source.power.min
    pmax = conf.source.power.max
    sig_min = conf.source.scale.min
    sig_max = conf.source.scale.max

    add_noise = conf.noise.add_noise
    snr_min = conf.noise.snr.min
    snr_max = conf.noise.snr.max

    kdistribution = conf.noise.kdistribution
    student = conf.noise.student
    mixture = conf.noise.mixture

    dof_min = conf.noise.dof.min
    dof_max = conf.noise.dof.max

    add_calibration_error = conf.calibration.add_calibration_error
    epsilon = conf.calibration.epsilon

    print(OmegaConf.to_yaml(conf))

    ms = MS(f"{ROOT_DIR}/../data/observation_profile/{msname}.MS")
    uvw = ms.uvw
    if coplanar:
        uvw[:,-1] = 0

    freq = ms.chan_freq.reshape(-1)

    nvis = len(uvw.reshape(-1,3))
    wl = speed_of_light / freq
    cellsize = np.min(wl)/(np.max(uvw)*2)

    model_images = zarr.zeros((nimage, npixel, npixel))

    clean_vis = zarr.zeros((nimage, nvis, len(freq)), dtype=complex)
    vis = zarr.zeros((nimage, nvis, len(freq)), dtype=complex)

    noise = zarr.zeros((nimage, nvis, len(freq)), dtype=complex)
    gains = zarr.zeros((nimage, nvis, len(freq)), dtype=complex)

    nsource = rng.integers(nsource_min, nsource_max, nimage)
    snr = rng.uniform(snr_min, snr_max, nimage)
    dof = rng.uniform(dof_min, dof_max, nimage)

    if from_sky_model:

        skymodel = np.load(sky_model_path)

        assert skymodel.shape[0] == npixel
        assert skymodel.shape[1] == npixel

        model_vis =  dirty2ms(
                            uvw = uvw,
                            freq = freq,
                            dirty = skymodel,
                            pixsize_x = cellsize,
                            pixsize_y = cellsize,
                            epsilon=1.0e-7
                        )

    for n, _ in enumerate(model_images):

        if from_sky_model:
            model_images[n,:,:] = skymodel
            clean_vis[n,:] = model_vis
        else:

            model_images[n,:,:] = random_skymodel(
                                            nsources=nsource[n],
                                            npixel=npixel,
                                            pmin=pmin,
                                            pmax=pmax,
                                            sig_min=sig_min,
                                            sig_max=sig_max,
                                            fixed=fixed,
                                            positions=positions,
                                            rng=rng
                                        )






            clean_vis[n,:] = dirty2ms(
                            uvw = uvw,
                            freq = freq,
                            dirty = model_images[n,:,:],
                            pixsize_x = cellsize,
                            pixsize_y = cellsize,
                            epsilon=1.0e-7
                        )


        P0 = np.linalg.norm(clean_vis[n,:,:]- np.mean(clean_vis[n,:,:]))**2 / nvis
        sigma2 = 10**(-snr[n]/10)*P0

        speckle = complex_normal(np.zeros_like(vis[n,:]), sigma2*np.eye(nvis), rng=rng)

        # Add noise and outliers
        if add_calibration_error:
            gains[n,:] = complex_normal(np.ones((nvis, len(freq))), epsilon**2 * np.eye(nvis))
            vis[n,:] =  gains[n,:] * clean_vis[n,:]
        else:
            vis[n,:] =  clean_vis[n,:]

        if add_noise:

            invgamma_texture = lambda dof: invgamma.rvs(dof/2, 0, dof/2, (nvis,1), random_state=rng)
            gamma_texture = lambda dof: gamma.rvs(dof, 0, 1/dof, (nvis,1), random_state=rng)
            inv_gauss_texture = lambda dof: invgauss.rvs(mu=1, loc=0, scale=1/dof, size=(nvis,1), random_state=rng)



            texture_distributions = (invgamma_texture, gamma_texture)
            # texture_distributions = (invgamma_texture, gamma_texture, inv_gauss_texture)
            if student:
                texture = invgamma_texture(dof[n])
                vis[n,:] =  vis[n,:] + texture*speckle
                noise[n,:] = texture*speckle
            elif kdistribution:
                texture = gamma_texture(dof[n])
                vis[n,:] =  vis[n,:] + texture*speckle
                noise[n,:] = texture*speckle
            elif mixture:

                d_idx = rng.integers(2)
                texture = texture_distributions[d_idx](dof[n])
                vis[n,:] =  vis[n,:] + texture*speckle
                noise[n,:] = texture*speckle
            else:
                vis[n,:] =  vis[n,:] + speckle
                noise[n,:] = speckle





    store = zarr.ZipStore(path, mode='w')
    root = zarr.group(store=store)
    data = root.create_group('data')
    data.create_dataset('vis', data=vis)
    data.create_dataset('noise', data=noise)
    data.create_dataset('gains', data=gains)
    data.create_dataset('clean_vis', data=clean_vis)
    data.create_dataset('model_images', data=model_images)
    info = root.create_group('info')
    info.create_dataset('uvw', data=uvw)
    info.create_dataset('freq', data=freq)

    now = datetime.datetime.now()
    metadata = {"created_at" : now.strftime("%Y-%m-%d %H:%M"),
                "config_file": configfile,
                "cellsize": cellsize,
                "nimage" : nimage,
                "npixel" : npixel}


    info.create_dataset('metadata', data=list(metadata.items()))
    store.close()

    with open(f'{name}.yaml', 'w') as file:
        file.write(OmegaConf.to_yaml(conf))
    return True


