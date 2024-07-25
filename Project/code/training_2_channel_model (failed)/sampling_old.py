import numpy as np
import torch 
import utils
from SongUnet import SongUNet
import os
import random
import yaml
import argparse
import cv2
from models.ncsnpp import NCSNpp
from config import get_config

# TRPA :- 

# from models.ncsnv2 import NCSNv2Deepest
# from condrefinenet import CondRefineNetDilated

def tensor_split(tensor, dim=1, each_channel=2):
    tensor_split = torch.split(tensor.unsqueeze(-1), each_channel, dim=dim)
    return torch.cat(tensor_split, dim=-1).mean(-1)


def denoiser(scorenet, sigma, x, c1, step_lr=0.9, c=4):
    sigma = torch.tensor(sigma).to(x.device)
    sigma = sigma.view(1,1,1,1)
    v_var = x.repeat(1,3,1,1)
    noise = torch.randn_like(v_var).clip(-1,1) * sigma * np.sqrt(2)
    
    inputs = v_var + noise

    logp = scorenet(inputs, sigma)
    v = x + c1 * tensor_split(logp*sigma**2)
    return v

def save_complex_images(Images, ImagePath):
    Images = Images.detach().cpu().numpy()
    AbsImages = np.abs(Images[:,0,:,:] + 1j * Images[:,1,:,:])
    for batchim in range(Images.shape[0]):
        NormImages = (AbsImages[batchim, :,:]/np.max(AbsImages[batchim, :,:]))*255
        NormImages = NormImages.astype(np.uint8)
        cv2.imwrite(ImagePath + str(batchim) + '.png', NormImages)


PATH_TO_TEST_DATA = ''
PATH_TO_LABEL_FILE_NAME = 'dataset.json'
PATH_TO_YAML_FILE = 'configs/file/brain_T2.yaml'


PATH_TO_TRAIN_DATA = './mri_data'
PATH_TO_MODELS_PATH = 'checkpoints'
MODEL_BASE_FILE_PATH = 'net.pth'

PATH_TO_AMBIENT_MODEL = '/home/akashp/Documents/mri-diffusion-training/models/SongNet_no_finetune_61.pt'

device = 'cuda'


# with open(PATH_TO_YAML_FILE) as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)

# def dict2namespace(langevin_config):
#     namespace = argparse.Namespace()
#     for key, value in langevin_config.items():
#         if isinstance(value, dict):
#             new_value = dict2namespace(value)
#         else:
#             new_value = value
#         setattr(namespace, key, new_value)
#     return namespace


# langevin_config = dict2namespace(config['langevin_config'])
# model = NCSNv2Deepest(langevin_config)
# model = model.to(device)
# states = torch.load(MODEL_BASE_FILE_PATH)[0]
# model_data = {}
# for key, value in states.items():
#     key_value = key.split('module.')[-1]
#     model_data[key_value] = value
# model.load_state_dict(model_data, strict=True)
# model.eval()



# TRPA :- 
'''
model = SongUNet(256, 4, 4, channel_mult = [1,1,1,1]).to(device)
model_data = {}
with open(PATH_TO_AMBIENT_MODEL, 'rb') as file:
    import pickle
    data = pickle.load(file)
for key, value in data['ema'].items():
    key = key.split('_orig_mod.model.')[-1].replace('384', '256').replace('192', '128').replace('96', '64').replace('48', '32').replace('24', '16')
    model_data[key] = value
model.load_state_dict(model_data)
# import pdb;pdb.set_trace()

sigma = 0.05
train_loader = utils.get_dataloader(3, 0)
image, label = next(train_loader)
noisy_image = image + np.random.randn(*image.shape) * sigma
noisy_image = torch.from_numpy(np.float32(noisy_image))

ones_mask = torch.ones_like(noisy_image).to(device)

noisy_image = noisy_image.to(device)
image = image.to(device)
save_complex_images(image, 'results/clean_image_')

with torch.inference_mode():
    model.eval()
    save_complex_images(noisy_image, 'results/noisy_image_')
    model_inp = torch.cat([noisy_image, ones_mask], axis = 1)
    # output_data = denoiser(model, sigma, model_inp, c)
    # import pdb; pdb.set_trace()
    for c in range(5,1000, 20):
        denoised_image = noisy_image + c*sigma**2*model(model_inp, torch.from_numpy(np.array([sigma])).to(device), None, None)[:,:2]  #*torch.ones((noisy_image.shape[0], 1, 1 ,1)).to(device)
        # import pdb; pdb.set_trace()
        save_complex_images(denoised_image, f'results/denoised_image_{c}_')
'''

config_file = get_config()

# Ambient-diffusion :- 
train_loader = utils.get_dataloader(3, 0)
image, label = next(train_loader)
model = NCSNpp(config_file).to(device)

import pdb;pdb.set_set_trace()
start = 1e-4


betas = torch.linspace(
start, end, noise_steps).to(device)
alphas = 1. - betas
alpha_hats = torch.cumprod(alphas, dim=0)

num_of_steps = 1000

with torch.inference_mode():
    n = 3
    label = None
    # x = torch.randn((n, 2, 256, 256)).to(device)
    for index in range(999, 0, -1):
        t = (torch.ones(n) * index).long().to(device)         
        pred_noise = model(x, t) # AT * grad(measurement - A*_) + model(_, t, label)
        alpha = alphas[t][:, None, None, None]
        alpha_hat = alpha_hats[t][:, None, None, None]
        beta = betas[t][:, None, None, None]
        if index > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise
        print(f'Iter {index}')
