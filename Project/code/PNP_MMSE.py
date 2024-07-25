import torch
import numpy as np
import sigpy as sp
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
import yaml
import cv2
import argparse
import h5py
import matplotlib.pyplot as plt


def dict2namespace(langevin_config):
    namespace = argparse.Namespace()
    for key, value in langevin_config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def get_mvue(kspace, s_maps):
    # convert from k-space to mvue
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=0) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=0))

def visualize(data, basename, normalization_value):
    import os
    data = data[:, 0] + 1j * data[:, 1]
    for index in range(data.shape[0]): 
        new_data = np.abs(data[index])
        new_data = new_data / np.max(new_data)
        image_d = np.array(255 * new_data)
        # plt.imsave(os.path.join('./results', f'{basename}_matplot_img{index}.png'), image_d, cmap='gray')
        cv2.imwrite(os.path.join('./results', f'{basename}_{normalization_value}_cv2_my_img{index}.png'), image_d)

def gradH(xtilde, x, ksp,  A, sigma, eta):
    return (1/sigma**2)*(A.H * A * xtilde - A.H * ksp) + (1/eta) * (xtilde - x)

def gradient_step(xtilde, x, ksp, A, alpha, graditer, sigma, eta):
    for iter in range(graditer):
        x = x - alpha*gradH(xtilde, x, ksp, A, sigma, eta)
    return x

def denoiser(img, sigma):
    pass
    

def PNPADMM(ksp, mps, mask, alpha, sigma, eta, maxiter, graditer):
    img_shape = mps.shape[1:]
        
    S = sp.linop.Multiply(img_shape, mps)
    F = sp.linop.FFT((ksp.shape[0], mask.shape[1], ksp.shape[2]), axes=(-1, -2))
    P = sp.linop.MatMul((ksp.shape[0], mask.shape[1], ksp.shape[2]), mask[np.newaxis, :, :])
    A = P * F * S

    xk = A.H * ksp
    vk = xk 
    uk = np.zeros_like(xk)

    for iter in range(maxiter):
        xk_1 = gradient_step(xk, vk - uk, ksp, A, alpha, graditer, sigma, eta)
        vk_1 = denoiser(xk_1+uk, sigma)
        uk_1 = uk + (xk_1 - vk_1)

        xk = xk_1
        vk = vk_1
        uk = uk_1

    return xk

def gradD(x, ksp, A):
    return (A.H * A * x - A.H * ksp)

def PNPFista(ksp, mps, mask, gamma, qk, sigma, maxiter):
    img_shape = mps.shape[1:]
        
    S = sp.linop.Multiply(img_shape, mps)
    F = sp.linop.FFT((ksp.shape[0], mask.shape[1], ksp.shape[2]), axes=(-1, -2))
    P = sp.linop.MatMul((ksp.shape[0], mask.shape[1], ksp.shape[2]), mask[np.newaxis, :, :])
    A = P * F * S

    xk = A.H * ksp
    sk = xk

    for iter in range(maxiter):
        zk_1 = sk - gamma*gradD(sk, ksp, A)
        # xk_1 = denoiser(zk_1, sigma)

        xk_1 = zk_1
        qk_1 = (1/2) * (1 + np.sqrt(1 + 4 * qk**2))
        
        sk_1 = xk_1 + ((qk_1-1)/qk) * (xk_1 - xk)

        sk = sk_1
        xk = xk_1
        qk = qk_1
    return xk

def main():
    path_to_checkpoint = './ncsnv2-mri-mvue/logs/mri-mvue/checkpoint_100000.pth'

    path_to_image_files = './test_images4/00000/img00000041.npy'
    config_file = './configs/file/brain_T2.yaml'
    h5_file_path1 = './datasets/brain_T2/file_brain_AXT2_200_2000019.h5'
    h5_file_path2 = './datasets/brain_T2_maps/file_brain_AXT2_200_2000019.h5'

    sigma_w = 0.04

    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)
            
    config_data = dict2namespace(config_data['langevin_config'])
    config_data.device = 'cuda'
    # model = NCSNv2Deepest(config_data).to(config_data.device)
    # model = torch.nn.DataParallel(model)
    # state_dict = torch.load(path_to_checkpoint)#, map_location=config_data.device)
    # model.load_state_dict(state_dict[0], strict=True)
    # model.eval()

    # Load h5 file
    ksp = np.load('ksp_data.npy')
    maps = np.load('maps_data.npy')

    norm_samples = np.load('./new.npy')

    # Generate mask
    theta1 = (np.pi/180)*5*np.random.rand(int(384/3))
    theta2 = (np.pi/180)*7*np.ones((int(384/3)))
    theta3 = (np.pi/180)*5*np.random.rand(int(384/3))

    a1 = np.multiply(np.multiply(np.sin(theta1), np.square(np.cos(theta2/2))), np.square(np.cos(theta3/2)))
    a2 = np.multiply(np.multiply(np.cos(theta1), np.sin(theta2)), np.square(np.cos(theta3/2)))
    a3 = np.multiply(np.multiply(np.cos(theta1), np.cos(theta2)), np.sin(theta3))

    aliasingmatrix = np.concatenate((np.diag(a1), np.diag(a2), np.diag(a3)), axis = 1)
    print(aliasingmatrix.shape)

    output = PNPFista(ksp, maps, aliasingmatrix, 0.0001, 1, 0.15, 100)

    print(output.shape)

if __name__ == '__main__':
    main()