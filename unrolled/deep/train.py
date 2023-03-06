from .models import *
from .datasets import ViDataset
from ..math.linalg import snr, psnr, nmse, ssim, normalized_cross_correlation

import click
from omegaconf import OmegaConf

import os 
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch import nn

import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")

def logprint(msg, path):
    with open(path, 'a') as file:
        print(msg, file=file)
    return True

def eval(model, dset_path, architecture, epoch=0, mode='test'):
    model.eval()
    test_set = ViDataset(dset_path)
    batch_size = 1

    dataloader = DataLoader(test_set, batch_size=batch_size)
    # loss = net.compute_loss(dataloader, nn.MSELoss(), device)
    size = len(dataloader.dataset)
    df = pd.DataFrame(columns=["model", "snr", "psnr", "nmse", "ssim", "ncc"])

    metrics = np.zeros((len(test_set), 5))

    for idx, (y, x) in enumerate(dataloader):
        y = y.to(device)

        # Compute prediction error
        estimated_image = model(y).detach().numpy().reshape(test_set.npixel, test_set.npixel)
        x = x.detach().numpy()
        true_image = x.reshape(test_set.npixel, test_set.npixel)

        curr_psnr = psnr(estimated_image, true_image)
        curr_ssim = ssim(estimated_image, true_image)
        curr_nmse = nmse(estimated_image, true_image)
        curr_snr = snr(estimated_image, true_image)
        curr_ncc = normalized_cross_correlation(estimated_image, true_image)

        metrics[idx] = np.array([curr_snr, curr_psnr, curr_nmse, curr_ssim, curr_ncc]).squeeze()


    df_mean = pd.DataFrame(np.mean(metrics, axis=0).reshape(1,metrics.shape[1]),
                    columns=['snr', 'psnr', 'nmse', 'ssim', 'ncc'])
    df_mean['reduce'] = 'mean'

    df_med = pd.DataFrame(np.median(metrics, axis=0).reshape(1,metrics.shape[1]),
                    columns=['snr', 'psnr', 'nmse', 'ssim', 'ncc'])
    df_med['reduce'] = 'median'


    df = pd.concat((df_mean, df_med), ignore_index=True)
    df['model'] = architecture
    df['dataset'] = mode
    df['epoch'] = epoch
    return df


def save_model(filename, nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out):
    return torch.save({
                    'epoch': nepoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dataset': dset_path,
                    'model name': model_name,
                    'loss': loss,
                    'uvw' : uvw,
                    'freq' : freq
                    }
                    , f'{out}/{filename}.pth')
         

@click.command()
@click.argument("config", type=click.Path(exists=True))
def train(config):

    conf = OmegaConf.load(config)
    model_name = conf.name
    nepoch = conf.nepoch
    step = conf.step
    batch_size = conf.batch_size

    depth = conf.depth
    dset_path = conf.dataset

    learning_rate = conf.learning_rate
    loss_type = conf.loss
    layers = conf.layers
    architecture = conf.architecture

    out_path = conf.out
    test_set_path = conf.test_set_path

    out = f'{out_path}/{model_name}.model'

    os.mkdir(out) if not os.path.exists(out) else print("Folder exists")

    tmp = dset_path.split('.')[0]
    shutil.copy(f'{tmp}.yaml', f'{out}/dataset.yaml')

# -------------------------------------------------------------------------------------

    dataset = ViDataset(dset_path)
    npixel = dataset.npixel
    uvw = dataset.uvw
    nvis = uvw.shape[0]
    freq = dataset.freq
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size)

    logpath = f"{out}/log.out"
    logprint(OmegaConf.to_yaml(conf), path=logpath)
    with open(f'{out}/{model_name}.yaml', 'w') as file:
        file.write(OmegaConf.to_yaml(conf))
# -------------------------------------------------------------------------------------


    if architecture == 'ROBUST':
        model = UnrolledRobust(depth, npixel, uvw, freq)
    elif architecture == 'GAUSS':
        model = Unrolled(depth, npixel, uvw, freq)
          




# -------------------------------------------------------------------------------------

    if loss_type == 'L2':
        loss_fn = nn.MSELoss()
    elif loss_type == 'L1':
        loss_fn = nn.L1Loss()


    df = pd.DataFrame()

    if layers == 'all':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(nepoch):
                        
            print(f"Epoch {epoch+1}\n-------------------------------")
            logprint(f"Epoch {epoch+1}\n-------------------------------", path=logpath)
            loss = model.train_supervised(train_dataloader, loss_fn, optimizer, device)
            logprint(f"loss = {loss}", path=logpath)

            if not epoch % step:
                save_model(f"{model_name}_ep-{epoch}", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)

                df_train = eval(model, dset_path, architecture, epoch=epoch, mode='train')
                df_test = eval(model, test_set_path, architecture, epoch=epoch, mode='test')
                df_epoch = pd.concat((df_test, df_train), ignore_index=True)
                df = pd.concat((df, df_epoch), ignore_index=True)
                logprint(f"metrics at epoch {epoch}", path=logpath)
                logprint(df_epoch, path=logpath)
                df.to_csv(f"{out}/metrics_tmp.csv", sep=';')

        save_model(f"{model_name}", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)

    elif layers == 'model':
        optimizer = torch.optim.Adam(model.trainable_model, lr=learning_rate)
        for epoch in range(nepoch):
                        
            print(f"Epoch {epoch+1}\n-------------------------------")
            logprint(f"Epoch {epoch+1}\n-------------------------------", path=logpath)
            loss = model.train_supervised(train_dataloader, loss_fn, optimizer, device)
            logprint(f"loss = {loss}", path=logpath)

            if not epoch % step:
                save_model(f"{model_name}_ep-{epoch}", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)
                
                df_train = eval(model, dset_path, architecture, epoch=epoch, mode='train')
                df_test = eval(model, test_set_path, architecture, epoch=epoch, mode='test')
                df_epoch = pd.concat((df_test, df_train), ignore_index=True)
                df = pd.concat((df, df_epoch), ignore_index=True)
                logprint(f"metrics at epoch {epoch}", path=logpath)
                logprint(df_epoch, path=logpath)
                df.to_csv(f"{out}/metrics_tmp.csv", sep=';')

        save_model(f"{model_name}", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)

    elif layers == 'robust':
        optimizer = torch.optim.Adam(model.trainable_robust, lr=learning_rate)
        for epoch in range(nepoch):
                        
            print(f"Epoch {epoch+1}\n-------------------------------")
            logprint(f"Epoch {epoch+1}\n-------------------------------", path=logpath)
            loss = model.train_supervised(train_dataloader, loss_fn, optimizer, device)
            logprint(f"loss = {loss}", path=logpath)

            if not epoch % step:
                save_model(f"{model_name}_ep-{epoch}", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)

                df_train = eval(model, dset_path, architecture, epoch=epoch, mode='train')
                df_test = eval(model, test_set_path, architecture, epoch=epoch, mode='test')

                df_epoch = pd.concat((df_test, df_train), ignore_index=True)
                df = pd.concat((df, df_epoch), ignore_index=True)
                logprint(f"metrics at epoch {epoch}", path=logpath)
                logprint(df_epoch, path=logpath)
                df.to_csv(f"{out}/metrics_tmp.csv", sep=';')

        save_model(f"{model_name}", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)

    else:
        pass
    
# -------------------------------------------------------------------------------------

    df["model name"] = model_name
    df["learning rate"] = learning_rate
    df["batch size"] = batch_size
    df["dataset name"] = dset_path
    df.to_csv(f"{out}/metrics.csv", sep=';')

    return True