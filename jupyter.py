#%%

import numpy as np
import matplotlib.pyplot as plt 

import zarr

from unrolled.source.astro.astro import robust_ml_imager

#%%
path = "/gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/datasets/stnoise_test.zip"


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

#%% 

from unrolled.source.math.linalg import ssim, snr, normalized_cross_correlation

idx = np.random.randint(0,len(images))
idx=35
print(idx)
model_image = images[idx]
estimated_image = robust_ml_imager(vis[idx], uvw, freq, cellsize, niter=100,
                 npix_x=32, npix_y=32, nu=5, alpha=0.0004,
                 gamma=0.001, miter=10)

plt.subplot(121)
plt.imshow(estimated_image)
plt.subplot(122)
plt.imshow(model_image)

err_snr = snr(estimated_image, model_image)
err_ssim = ssim(estimated_image, model_image)
err_ncc = normalized_cross_correlation(estimated_image, model_image)
print(f"snr = {err_snr}\nssim = {err_ssim}\nncc = {err_ncc}")

# %%
import torch
from torchvision.transforms import ToTensor

from torch import nn

from unrolled.source.deep.models import Unrolled

#%%

net = Unrolled(depth=10, npixel=32, uvw=uvw, freq=freq, robust=True)
baseline_net = Unrolled(depth=10, npixel=32, uvw=uvw, freq=freq, robust=True)

model_path = "/gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/models/bench/unrolled-all/unrolled-all.pth"
model_path = "/gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/models/kdistribution-A/kdistribution-A.pth"
model_path = "/gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/models/kdistribution/unrolled-robust/unrolled-robust.pth"
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint["model_state_dict"])

# model_path = "/gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/models/lr=1e-4/unrolled-robust/unrolled-robust.pth"
# checkpoint = torch.load(model_path)
# baseline_net.load_state_dict(checkpoint["model_state_dict"])

#%% 
W0 = net.robust_layer[0].bias.detach().numpy()
W1 = baseline_net.robust_layer[0].bias.detach().numpy()

print(np.sum((W0 - W1)**2))


# %%

path = "/gpfs/users/mhiriy/Documents/EUSIPCO-23/results/examples.zip"


store = zarr.ZipStore(path)
z = zarr.group(store=store)

for key in z:
    plt.figure()
    plt.title(key)
    plt.imshow(z[key][0],cmap='nipy_spectral')
    plt.colorbar()
# %%

from unrolled.source.astro.ms import MS

mspath = "/gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/observation_profile/vla.MS"
mspath = "/gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/observation_profile/random_static.MS"
ms = MS(mspath)

uvw = ms.uvw

print(len(uvw))
# %%

m =nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding='same'),
            nn.AdaptiveMaxPool1d(output_size=16),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.ReLU(),
            )   
n = nn.Sequential(    
            nn.Linear(16, 1),
            nn.ReLU()
    )


a = torch.randn((48,1, 64))
b = nn.AdaptiveMaxPool1d(output_size=16)(a)
output = n(torch.transpose(m(a)))
# %%
