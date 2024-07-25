from utils import Diffusion
import torch
import numpy as np
import cv2

def visualizer_pdata(path_to_data):
    data = np.load(path_to_data)
    batch = data.shape[0]
    import pdb;pdb.set_trace()
    for index in range(batch):
        batch_data = data[index]
        flipped_img_data = batch_data[0] + 1j * batch_data[1]
        final_numpy_data_viz = np.array(255*np.abs(flipped_img_data)).astype(np.uint8)
        viz_filename = f"ptest_{index}.png"
        cv2.imwrite(viz_filename, final_numpy_data_viz)

def visualizer_data(data):
    batch = data.shape[0]
    for index in range(batch):
        batch_data = data[index]
        flipped_img_data = batch_data[0] + 1j * batch_data[1]
        final_numpy_data_viz = np.array(255*np.abs(flipped_img_data)).astype(np.uint8)
        viz_filename = f"test_{index}.png"
        cv2.imwrite(viz_filename, final_numpy_data_viz)

# visualizer_pdata('/home/akashp/Documents/mri-diffusion-training/image_gen.npy')

data_device = torch.device('cuda')
diffusion = Diffusion(1000, None, c_in= 2, c_out=2, device=data_device)
diffusion.load_weights('models/SongNet_no_finetune_61.pt')

start = 1e-4
end = 0.02
noise_steps = 5000

betas = torch.linspace(
start, end, noise_steps).to(data_device)
alphas = 1. - betas
alpha_hats = torch.cumprod(alphas, dim=0)

with torch.inference_mode():
    n = 3
    label = None
    x = torch.randn((n, 2, 256, 256)).to(data_device)
    for index in range(noise_steps - 1, 0, -1):
        t = (torch.ones(n) * index).long().to(data_device)   
        sigma = t.reshape(n, 1, 1, 1)      
        # t = t.reshape(n, 1, 1, 1)
        # import pdb;pdb.set_trace()
        pred_noise = diffusion.model(x, sigma[:, 0, 0, 0], None, None) # AT * grad(measurement - A*_) + model(_, t, label)
        # net(x_tilda, self.sigma_w[:,0,0,0], labels, augment_labels=augment_labels)
        alpha = alphas[t][:, None, None, None]
        alpha_hat = alpha_hats[t][:, None, None, None]
        beta = betas[t][:, None, None, None]
        if index > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise
        print(f'Iter {index}')
    visualizer_data(x.detach().cpu().numpy())
