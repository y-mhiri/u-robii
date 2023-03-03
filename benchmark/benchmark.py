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

def save_img_to_zarr(name, z, images, vis, func, *args, **kwargs):

    outputs = np.zeros_like(images)
    for idx,vi in enumerate(vis):
        estimated_image = func(vi, *args, **kwargs)
        outputs[idx] = estimated_image

    z.create_dataset(name, data=outputs)

    return z
    
def bench(name, images, vis, func, *args, **kwargs):
    nimage = len(images)
    metrics = np.zeros((nimage, 5))

    for idx,vi in enumerate(vis):

        estimated_image = func(vi, *args, **kwargs)

        curr_psnr = psnr(estimated_image, images[idx])
        curr_ssim = ssim(estimated_image, images[idx])
        curr_nmse = nmse(estimated_image, images[idx])
        curr_snr = snr(estimated_image, images[idx])
        curr_ncc = normalized_cross_correlation(estimated_image, images[idx])

        metrics[idx] = np.array([curr_snr, curr_psnr, curr_nmse, curr_ssim, curr_ncc]).squeeze()


    df = pd.DataFrame(np.mean(metrics, axis=0).reshape(1,metrics.shape[1]),
                      columns=['snr', 'psnr', 'nmse', 'ssim', 'ncc'],
                      index=[name])
    return df


def main():

    test_set_path = '/gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/datasets/stnoise_test.zip'
    out_path = '/gpfs/users/mhiriy/Documents/EUSIPCO-23/results'
    model_path = '/gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/models/bench'



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

    store_out = zarr.ZipStore(f"{out_path}/examples-st-unrolled.zip", mode='w')    
    z_out = zarr.group(store=store_out)
    
    noutput_image = 100
    idxs = np.random.randint(0, len(images), noutput_image)
    images_np = np.array(images).squeeze()
    vis_np = np.array(vis).squeeze()
    selected_images = images_np[idxs]
    selected_vis = vis_np[idxs]
    z_out.create_dataset("model_images", data=selected_images)
    


    # EM parameters

    
    df = pd.DataFrame(columns=["snr", "psnr", "nmse", "ssim", "ncc"])

    # metrics = bench("ML Imager - threshold = 0.01", selected_images, selected_vis, robust_ml_imager, 
    #         uvw=uvw, 
    #         freq=freq, 
    #         cellsize=cellsize, 
    #         niter=niter,
    #         npix_x = npix_x, 
    #         npix_y = npix_y,
    #         alpha=0.01,
    #         gamma=0.01,
    #         miter=1,
    #         nu=5
    #         )

    # z_out = save_img_to_zarr("ML Imager - 0.01", z_out, selected_images, selected_vis, robust_ml_imager, 
    #         uvw=uvw, 
    #         freq=freq, 
    #         cellsize=cellsize, 
    #         niter=niter,
    #         npix_x = npix_x, 
    #         npix_y = npix_y,
    #         alpha=0.01,
    #         gamma=0.01,
    #         miter=1
    #         )

    # df = df.append(metrics)
    # print(df)


    metrics = bench("ML Imager - 0.05", selected_images, selected_vis, robust_ml_imager, 
            uvw=uvw, 
            freq=freq, 
            cellsize=cellsize, 
            niter=10,
            npix_x = npix_x, 
            npix_y = npix_y,
            alpha=0.0005,
            gamma=0.0001,
            miter=10,
            nu=5
            )

    z_out = save_img_to_zarr("ML Imager - 0.05", z_out, selected_images, selected_vis, robust_ml_imager, 
            uvw=uvw, 
            freq=freq, 
            cellsize=cellsize, 
            niter=50,
            npix_x = npix_x, 
            npix_y = npix_y,
            alpha=0.0005,
            gamma=0.0001,
            miter=10,
            nu=5
            )

    # df = df.append(metrics)
    # print(df)

    # metrics = bench("ML Imager - alpha = 0.025", images, vis, robust_ml_imager, 
    metrics = bench("EM Imager", images, vis, robust_ml_imager, 
            uvw=uvw, 
            freq=freq, 
            cellsize=cellsize, 
            niter=100,
            npix_x = npix_x, 
            npix_y = npix_y,
            alpha=0.0005,
            gamma=0.001,
            miter=10,
            nu=5
            )
    
    z_out = save_img_to_zarr("EM Imager ", z_out, selected_images, selected_vis, robust_ml_imager, 
            uvw=uvw, 
            freq=freq, 
            cellsize=cellsize, 
            niter=100,
            npix_x = npix_x, 
            npix_y = npix_y,
            alpha=0.0005,
            gamma=0.001,
            miter=10,
            nu=5
            )
    
    df = df.append(metrics)
    print(df)


    depth = 10
    alpha = None
    robust = True

    for filename in os.listdir(model_path):


        checkpoint_path = f"{model_path}/{filename}/{filename}.pth"

        metrics = bench(f"Unrolled ML Imager - {filename}", images, vis, eval_net, 
                checkpoint_path, 
                depth=depth,
                alpha=alpha,
                npixel=int(metadata["npixel"]),
                uvw=uvw,
                freq=freq, 
                robust=robust
                )


        z_out = save_img_to_zarr(f"Unrolled - {filename}", z_out, selected_images, selected_vis, eval_net, 
                checkpoint_path, 
                depth=depth,
                alpha=alpha,
                npixel=int(metadata["npixel"]),
                uvw=uvw,
                freq=freq, 
                robust=robust
                )


        df = df.append(metrics)
        print(df)



#     metrics = bench(f"CLEAN", images, vis, clean, 
#              gamma=0.1,
#              cellsize = cellsize,
#              npix_x=int(metadata["npixel"]),
#              npix_y=int(metadata["npixel"]),
#              uvw=uvw,
#              freq=freq, 
#              niter=10, 
#              )
#     df = df.append(metrics)
#     print(df)

    df.to_csv(f"{out_path}/metrics-st-new.csv")

    return True


if __name__ == '__main__':
    main()

 