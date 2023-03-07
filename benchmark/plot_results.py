import matplotlib.pyplot as plt
import pandas as pd
import zarr
import numpy as np

from unrolled.math.linalg import ssim, snr, normalized_cross_correlation
from unrolled.astro.robust_ml_imager import robust_ml_imager, unrolledImager

import click

def compute_errors(estimated_image, true_image):
    err_snr = snr(estimated_image, true_image)
    err_ssim = ssim(estimated_image, true_image)
    err_ncc = normalized_cross_correlation(estimated_image, true_image)  

    return err_snr, err_ssim, err_ncc


def import_dataset_from_zarr(path):

    store = zarr.ZipStore(path)
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

    return images, vis, uvw, freq, cellsize, npix_x, npix_y, z




@click.command()
@click.argument('dsetpath', type=click.Path(exists=True) )
@click.argument('modelpath', type=click.Path(exists=True) )
@click.argument('out', type=click.Path(exists=True) )
def main(dsetpath, modelpath, out):

    # path = "/synced/data/datasets/student_test.zip"
    # model_path = "/synced/data/student-0.0001.model/student-0.0001.pth"
    # out ='.'

    images, vis, uvw, freq, cellsize, npix_x, npix_y, z = import_dataset_from_zarr(dsetpath)

    def show_image_examples(vis, images, model_path, idx='random', save=True, path='.'):
        if idx == 'random':
            idx = np.random.random_integers(len(images))
        else: 
            pass
        
        estimated_image = unrolledImager(vis[idx], model_path, freq, uvw, npix_x, npix_y, cellsize, niter=10)
        estimated_image_em =  robust_ml_imager(vis[idx], uvw, freq, cellsize, niter=100,
                        npix_x=npix_x, npix_y=npix_y, nu=5, alpha=0.0004,
                        gamma=0.001, miter=10)

        cmap = "Spectral_r"


        plt.figure(figsize=(20,5))

        plt.subplot(131)
        plt.title('Unrolled robust')
        plt.imshow(estimated_image, cmap=cmap)
        plt.colorbar()


        plt.subplot(132)
        plt.title('EM student-t')
        plt.imshow(estimated_image_em, cmap=cmap)
        plt.colorbar()


        plt.subplot(133)
        plt.title('True image')
        plt.imshow(images[idx], cmap=cmap)
        plt.colorbar()

        if save:
            plt.savefig(path)
        else:        
            plt.show()


    def benchmark(vis, images, model_path, save=False, path='.'):

        df = pd.DataFrame(columns=['snr', 'ssim', 'ncc', 'method', 'image_idx'])

        for idx, (vi, true_image) in enumerate(zip(vis, images)):
            estimated_image_unrolled = unrolledImager(vi, model_path, freq, uvw, npix_x, npix_y, cellsize, niter=10)
            estimated_image_em =  robust_ml_imager(vi, uvw, freq, cellsize, 
                                                    npix_x=npix_x, npix_y=npix_y, 
                                                    nu=5, alpha=0.0004,
                                                    niter=100, gamma=0.001, miter=10)

            err_snr, err_ssim, err_ncc = compute_errors(estimated_image_unrolled, true_image)
            df_unrolled = pd.DataFrame(np.array([[err_snr, err_ssim, err_ncc]]), columns=['snr','ssim', 'ncc'])
            df_unrolled['method'] = 'Unrolled Robust'
            df_unrolled['image_idx'] = idx
            err_snr, err_ssim, err_ncc = compute_errors(estimated_image_em, true_image)
            df_em = pd.DataFrame(np.array([[err_snr, err_ssim, err_ncc]]), columns=['snr','ssim', 'ncc'])
            df_em['method'] = 'Robust EM'
            df_em['image_idx'] = idx

            df = pd.concat((df, df_em), ignore_index=True)
            df = pd.concat((df, df_unrolled), ignore_index=True)

        if save:
            df.to_csv(path)

        return df



    for ii in range(10):
        show_image_examples(vis, images, modelpath, idx='random', save=True, path=f'{out}/images-{ii}.png')

    df = benchmark(vis, images, modelpath, save=True, path=f'{out}/metrics.csv')
    fig, axes = plt.subplots(1,3,sharex=False,sharey=False)
    df.boxplot(by='method', ax=axes, column=['ncc', 'snr', 'ssim'], 
                showfliers=False, fontsize=7)

    plt.savefig(f'{out}/plot.png')

    return True




if __name__ == '__main__':
    main()