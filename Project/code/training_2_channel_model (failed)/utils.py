import torch.nn as nn
import torch
from torch.optim import Adam
import torch.nn.functional as F
import os
from SongUnet import SongUNet
from loss import SureLoss
import random
import numpy as np
import json
import sigpy as sp
from tqdm import tqdm

PATH_TO_TRAIN_DATA = './mri_data'
PATH_TO_MODELS_PATH = 'models'
MODEL_BASE_FILE_PATH = 'SongNet_no_finetune'
PATH_TO_LABEL_FILE_NAME = 'dataset.json'


def set_seed(s=41, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try:
        torch.manual_seed(s)
    except NameError:
        pass
    try:
        torch.cuda.manual_seed_all(s)
    except NameError:
        pass
    try:
        np.random.seed(s % (2**32-1))
    except NameError:
        pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_mini_train_data(path_to_train_files, noise_value):
    mini_batch_train_data = []
    for filename in path_to_train_files:
        # print(filename)
        full_train_file_path = os.path.join(PATH_TO_TRAIN_DATA, filename)
        with open(full_train_file_path, 'rb') as f:
            image = np.load(f)
        image = sp.resize(image, (2, image.shape[1], image.shape[1]))
        noisy_image = image + np.random.randn(*image.shape) * noise_value
        noisy_image = np.float32(noisy_image)
        mini_batch_train_data.append(noisy_image)

    return torch.from_numpy(np.array(mini_batch_train_data))


def get_mini_label_data(label_data):
    return torch.from_numpy(np.array([int(value[1]) for value in label_data]))


def get_dataloader(batch_size, noise_value_to_add):
    all_fnames = {os.path.relpath(os.path.join(root, fname), start=PATH_TO_TRAIN_DATA)
                  for root, _dirs, files in os.walk(PATH_TO_TRAIN_DATA) for fname in files}
    all_fnames.remove(PATH_TO_LABEL_FILE_NAME)
    all_fnames = np.array(sorted(all_fnames))
    label_file_path = os.path.join(PATH_TO_TRAIN_DATA, PATH_TO_LABEL_FILE_NAME)

    with open(label_file_path) as file:
        label_data = np.array(json.load(file)['labels'])
    length_train_data, length_label_data = len(all_fnames), len(label_data)
    assert length_train_data == length_label_data, "Failed because train and label length don't match"
    index_data = np.arange(length_train_data)
    np.random.shuffle(index_data)
    length_train_data = 5000
    for index_value in range(0, length_train_data, batch_size):
        index_values = index_data[index_value:index_value+batch_size]
        mini_x = get_mini_train_data(
            all_fnames[index_values], noise_value_to_add)
        mini_y = get_mini_label_data(label_data[index_values])
        yield mini_x, mini_y


def dict2namespace(langevin_config):
    namespace = argparse.Namespace()
    for key, value in langevin_config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class Diffusion:
    def __init__(self, noise_steps, path_to_config, start=1e-4, end=0.02, img_size=256, num_classes=None, c_in=2, c_out=3, device='cpu', batch_size=32, noise_level=0):
        self.noise_steps = noise_steps
        self.start = start
        self.end = end
        self.img_size = img_size
        self.num_classes = num_classes
        self.c_in = c_in
        self.c_out = c_out
        self.device = device
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.loss = SureLoss()

        self.betas = torch.linspace(
            start, end, self.noise_steps).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # with open(path_to_config) as f:
        #     self.config = yaml.load(f, Loader=yaml.FullLoader)

        # self.langevin_config = dict2namespace(self.config['langevin_config'])
        # NCSNv2Deepest(self.langevin_config)
        self.model = SongUNet(self.img_size, self.c_in, self.c_out)
        self.model = self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-5)

    def sample_timestamp(self, batch_size):
        return torch.randint(high=self.noise_steps, size=(batch_size, ), device=self.device).long()

    def get_noisy_images(self, X, t):
        noise = torch.randn_like(X).to(self.device)
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[
            :, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(
            1 - self.alphas_cumprod[t])[:, None, None, None]
        return sqrt_alphas_cumprod_t * X + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def load_weights(self, path_to_pretrained_model):
        states = torch.load(path_to_pretrained_model)
        self.model.load_state_dict(states)
        # optim_states = torch.load(path_to_optimizer)
        # self.optimizer.load_state_dict(optim_states)
        # print('all ok?')
        # import pdb;pdb.set_trace()        
        # model_data = {}
        # # removing module because we are not using the langenvinoptimizer which sits on top of the model.
        # for key, value in states.items():
        #     key_value = key.split('module.')[-1]
        #     model_data[key_value] = value
        # self.model.load_state_dict(model_data, strict=True)

    def save_weights(self, epoch_value):
        os.makedirs(PATH_TO_MODELS_PATH, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(
            PATH_TO_MODELS_PATH, f'{MODEL_BASE_FILE_PATH}_{epoch_value}.pt'))
        torch.save(self.optimizer.state_dict(), os.path.join(
            PATH_TO_MODELS_PATH, f"{MODEL_BASE_FILE_PATH}__optim_{epoch_value}.pt"))

    def get_loss_value(self, model, actual):
        return self.loss(model, actual)

    def one_epoch(self, is_train_epoch=True):
        loss = 0
        count = 0
        if is_train_epoch:
            self.model.train()
        else:
            self.model.eval()
        train_loader = get_dataloader(self.batch_size, self.noise_level)
        for images, labels in tqdm(train_loader):
            if is_train_epoch:
                self.optimizer.zero_grad()

            images = images.to(self.device)
            # labels = labels.to(self.device)
            # t = self.sample_timestamp(images.shape[0])
            # noisy_images, noise = self.get_noisy_images(images, t)
            # predicted_noise = self.model(noisy_images, t, labels)
            # loss_value = self.get_loss_value(noise, predicted_noise)
            loss_value = self.get_loss_value(self.model, images)
            loss += loss_value
            if is_train_epoch:
                loss_value.backward()
                self.optimizer.step()

        return loss.mean().item()

    def sample(self):
        # tweedie denoising
        # get sigma value
        normalizing_constant = 1
        t = self.sample_timestamp(images.shape[0])

        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[
            :, None, None, None]  # Assuming sigma values are these, need to clear this
        Denoised_images = noisy_images + \
            sqrt_alphas_cumprod_t*self.model(noisy_images)

        pass

    def log_data(self):
        self.sample()
        pass

    def train(self, epochs):
        for epoch in range(47, epochs):

            loss_train = self.one_epoch()
            print(f'{epoch=}, {loss_train=}')  # , {loss_val = }')

            # loss_val = self.one_epoch(False)

            if epoch % 100 == 0:
                # sample
                # send some noisy images (maybe from validation or buffer few?) and check if it's denoising with increase in epochs.
                # self.log_data()
                pass

            self.save_weights(epoch)
