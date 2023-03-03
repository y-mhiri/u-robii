#%%
import os
import numpy  as np 
import zarr
import pandas as pd 

from unrolled.source.deep.models import *
from unrolled.source.deep.datasets import ViDataset

from unrolled.source.math.linalg import snr, psnr, ssim, nmse, normalized_cross_correlation

from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")

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
nimage = len(images)

vis = z["data/vis"]
uvw = np.array(z["info/uvw"]).squeeze()
freq = np.array(z["info/freq"]).reshape((1))
cellsize = float(metadata['cellsize'])

npix_x = int(metadata["npixel"])
npix_y = int(metadata["npixel"])


depth = 10
net = Unrolled(depth, npix_x, uvw, freq, robust=True, alpha=None)


path_to_model = "/gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/models/lr=1e-4/unrolled-model/unrolled-model.pth"
path_to_robust = "/gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/models/lr=1e-4/unrolled-robust/unrolled-robust.pth"
path_all = "/gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/models/lr=1e-4/unrolled.pth"

checkpoint_model = torch.load(path_to_model)
checkpoint_robust = torch.load(path_to_robust)
checkpoint_robust["model_state_dict"]["W.weight"] = checkpoint_model["model_state_dict"]["W.weight"]
# print(checkpoint_robust["model_state_dict"]["W.weight"])
net.load_state_dict(checkpoint_robust["model_state_dict"])

torch.save(checkpoint_robust, path_all)
# %%
dataset = ViDataset(test_set_path)
batch_size = 1

dataloader = DataLoader(dataset, batch_size=batch_size)
# loss = net.compute_loss(dataloader, nn.MSELoss(), device)
size = len(dataloader.dataset)
df = pd.DataFrame(columns=["snr", "psnr", "nmse", "ssim", "ncc"])

metrics = np.zeros((nimage, 5))

for idx, (y, x) in enumerate(dataloader):
    y, x = y.to(device), x.to(device)

    # Compute prediction error
    estimated_image = net(y).detach().numpy().reshape(npix_x, npix_y)


    curr_psnr = psnr(estimated_image, images[idx])
    curr_ssim = ssim(estimated_image, images[idx])
    curr_nmse = nmse(estimated_image, images[idx])
    curr_snr = snr(estimated_image, images[idx])
    curr_ncc = normalized_cross_correlation(estimated_image, images[idx])

    metrics[idx] = np.array([curr_snr, curr_psnr, curr_nmse, curr_ssim, curr_ncc]).squeeze()


df = pd.DataFrame(np.mean(metrics, axis=0).reshape(1,metrics.shape[1]),
                    columns=['snr', 'psnr', 'nmse', 'ssim', 'ncc'],
                    index=["Unfolded"])
    
print(df)


