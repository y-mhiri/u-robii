import os
import numpy  as np 
import zarr
import pandas as pd 

from omegaconf import OmegaConf
# from unrolled.imager.robust_imager import rem_imager_vis_domain, rem_imager_fourier_domain
# from unrolled.imager.clean import clean
from unrolled.source.astro.astro import robust_ml_imager
from unrolled.source.math.linalg import snr, psnr, ssim, nmse, normalized_cross_correlation
from unrolled.source.deep.models import *

from multiprocessing import Pool 
import multiprocessing as mp
from functools import partial

global ncore 
ncore = mp.cpu_count()

def save_img_to_zarr(name, z, images, vis, func, *args, **kwargs):

    outputs = np.zeros_like(images)
    for idx,vi in enumerate(vis):
        estimated_image = func(vi, *args, **kwargs)
        outputs[idx] = estimated_image

    z.create_dataset(name, data=outputs)

    return z
    

def compute_error(vi, image, func, **kwargs):
    estimated_image = func(vi, **kwargs)

    curr_psnr = psnr(estimated_image, image)
    curr_ssim = ssim(estimated_image, image)
    curr_nmse = nmse(estimated_image, image)
    curr_snr = snr(estimated_image, image)
    curr_ncc = normalized_cross_correlation(estimated_image, image)

    metrics = np.array([curr_snr, curr_psnr, curr_nmse, curr_ssim, curr_ncc]).squeeze()
    return estimated_image, metrics


def bench(name, images, vis, func, *args, **kwargs):
    nimage = len(images)
    # metrics = np.zeros((nimage, 5))


    with Pool(ncore - 1) as p:   
        res = p.starmap(partial(compute_error, func=func, **kwargs), zip(list(vis), list(images)))

    metrics = np.array([r[1] for r in res])

    # for idx,vi in enumerate(vis):

    #     estimated_image = func(vi, *args, **kwargs)

    #     curr_psnr = psnr(estimated_image, images[idx])
    #     curr_ssim = ssim(estimated_image, images[idx])
    #     curr_nmse = nmse(estimated_image, images[idx])
    #     curr_snr = snr(estimated_image, images[idx])
    #     curr_ncc = normalized_cross_correlation(estimated_image, images[idx])

    #     metrics[idx] = np.array([curr_snr, curr_psnr, curr_nmse, curr_ssim, curr_ncc]).squeeze()


    df = pd.DataFrame(np.mean(metrics, axis=0).reshape(1,metrics.shape[1]),
                      columns=['snr', 'psnr', 'nmse', 'ssim', 'ncc'],
                      index=[name])
    return df


def main():

    test_set_path = '/gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/datasets/stnoise_test.zip'
    out_path = '/gpfs/users/mhiriy/Documents/EUSIPCO-23/results'
    model_path = '/gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/models/bench'
    logpath = '/gpfs/users/mhiriy/Documents/EUSIPCO-23/results/benchmark-em.log'


    store = zarr.ZipStore(test_set_path)    
    z = zarr.group(store=store)

    metadata = dict(z["info/metadata"])
    print(
        f"Dataset info : \n \
        created_at : {metadata['created_at']} \n \
        number of image : {metadata['nimage']} \n \
        image size : {metadata['npixel']}x{metadata['npixel']} pixels \n \
        cellsize : {metadata['cellsize']} rad"
    )  

    images = z["data/model_images"]
    vis = z["data/vis"]
    uvw = np.array(z["info/uvw"]).squeeze()
    freq = np.array(z["info/freq"]).reshape((1))
    cellsize = float(metadata['cellsize'])

    npix_x = int(metadata["npixel"])
    npix_y = int(metadata["npixel"])

    store_out = zarr.ZipStore(f"{out_path}/examples-st-em.zip", mode='w')    
    z_out = zarr.group(store=store_out)
    
    noutput_image = 100

    rng = np.random.default_rng(0)
    idxs = rng.integers(0, len(images), noutput_image)
    images_np = np.array(images).squeeze()
    vis_np = np.array(vis).squeeze()
    selected_images = images_np[idxs]
    selected_vis = vis_np[idxs]
    z_out.create_dataset("model_images", data=selected_images)
    


    # EM parameters

    
    df = pd.DataFrame(columns=["snr", "psnr", "nmse", "ssim", "ncc"])

    with open(logpath,'w') as file:
        print("Starting ...", file=file)
        
    for niter in np.arange(10,110,10):

        metrics = bench(f"EM Imager - niter = {niter}", images, vis, robust_ml_imager, 
                uvw=uvw, 
                freq=freq, 
                cellsize=cellsize, 
                niter=niter,
                npix_x = npix_x, 
                npix_y = npix_y,
                alpha=0.00025,
                gamma=0.0001,
                miter=10,
                nu=5
                    )
        
        # z_out = save_img_to_zarr(f"EM Imager - niter = {niter}", z_out, selected_images, selected_vis, robust_ml_imager, 
        #         uvw=uvw, 
        #         freq=freq, 
        #         cellsize=cellsize, 
        #         niter=niter,
        #         npix_x = npix_x, 
        #         npix_y = npix_y,
        #         alpha=0.00025,
        #         gamma=0.0001,
        #         miter=10,
        #         nu=5
        #         )
        
        df = df.append(metrics)

        with open(logpath,'w') as file:
            print(df, file=file)
        

    df.to_csv(f"{out_path}/metrics-st-em-mp.csv")

    return True


if __name__ == '__main__':
    main()

 