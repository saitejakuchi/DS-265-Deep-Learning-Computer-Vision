import os
import torch
import networks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from renderer import render_seq
from tqdm import tqdm
from itertools import zip_longest
from torch.utils.data import DataLoader

PATH_TO_MOTION_DATASET = './data/uptown_funk.json'
PATH_TO_MODELS = './models'
PATH_TO_RESULT_FILES = './results'
PATH_TO_CONFIG_FILE = 'config.yaml'


def save_plot_data(plot_data, label_data, save_file_name):
    '''
    Generate loss plots
    '''
    fig = plt.figure(figsize=(15, 15))
    for plot, label in zip(plot_data, label_data):
        plt.plot(plot, label=label)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    title = 'Training Loss Curve'
    if len(plot_data) > 1:
        title += 's'
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(PATH_TO_RESULT_FILES,
                save_file_name))
    plt.close()


def convert_to_json_format(columns_data, actual_data):
    '''
    Convert array data to json data so that it can be passed to render python file.
    '''
    tensor_data = actual_data.permute(2, 0, 1)
    render_json = {}
    for index, key in enumerate(columns_data):
        render_json[key] = tensor_data[:, index, :].cpu().numpy().tolist()
    return render_json


class MotionDataLoader:
    def __init__(self) -> None:
        '''
        Read the data for further processing.
        '''
        dataframe = pd.read_json(PATH_TO_MOTION_DATASET)
        self.columns = dataframe.columns
        self._normalize_data(dataframe)

    def _normalize_data(self, data):
        '''
        normalize the data accordingly.
        '''
        self.motion_data = np.array(data.values.tolist())
        self.min_x, self.max_x, self.min_y, self.max_y = np.min(self.motion_data[:, :, 0]), np.max(
            self.motion_data[:, :, 0]), np.min(self.motion_data[:, :, 1]), np.max(self.motion_data[:, :, 1])
        self.motion_data[:, :, 0] = (
            self.motion_data[:, :, 0] - self.min_x) / (self.max_x - self.min_x)
        self.motion_data[:, :, 1] = (
            self.motion_data[:, :, 1] - self.min_y) / (self.max_y - self.min_y)
        # print(f'{self.min_x = } {self.max_x = } {self.min_y = } {self.max_y = }')

    def get_motion_dataloader(self, batch_size, N, overlap):
        '''
        Create a dataloader of the above normalized data based on the batch size, overlap and N value passed.
        '''
        total_frames = self.motion_data.shape[0]
        train_data = None
        if overlap:
            train_data = np.array([list(self.motion_data[index:index+N])
                                   for index in range(total_frames - N + 1)])
        else:
            seq_batch_count = total_frames // N
            train_data = np.array([list(group) for group in zip_longest(
                *[iter(self.motion_data)] * N)][:seq_batch_count])
        # transpose for better use. (new_frame_count, columns_count, 2, N)
        train_data = train_data.transpose(0, 2, 3, 1).astype(np.float32)
        train_data = torch.tensor(train_data)
        print(f'{train_data.shape =  }')
        assert train_data.shape[-1] == N, "Error in tranposing final axis of data"
        assert train_data.shape[-2] == 2, "Error in tranposing pre-final axis of data"
        return DataLoader(train_data, batch_size, shuffle=True)


class TrainingLoop:
    def __init__(self, data_loader, configuration_data):
        self.data_loader = data_loader
        for k, v in configuration_data.items():
            setattr(self, k, v)

        self.overlap_str = 'nonoverlap'
        if self.overlap:
            self.overlap_str = 'overlap'

        self.total_frames = None
        self.train_dataloader = None

        self.motion_args = {'num_heads': 4, 'cond_mode': '',
                            'cond_mask_prob': 0, 'dataset': 'json'}

        self.punet_model = networks.MDM_UNetModel(
            self.motion_args, self.batch_size, 40, 32, 40, 4, 2, 0.1)
        self.punet_model = self.punet_model.to(self.device)

        self.diffusion = networks.DDPM(
            self.punet_model, self.noise_steps, self.beta_start, self.beta_end, self.device)
        self.loss_values = []

        if not self.is_train:
            self._load_weights()

    def _load_weights(self):
        self.punet_model.load_state_dict(
            torch.load(self.path_to_trained_model))

    def train(self):
        '''
        Train the MDM_UNet using Diffusion on MSE Loss.
        '''
        train_dataloader = self.data_loader.get_motion_dataloader(
            self.batch_size, self.N, self.overlap)
        optimizer = torch.optim.AdamW(
            self.punet_model.parameters(), lr=self.learning_rate)

        for epoch in tqdm(range(self.epochs), total=self.epochs):
            epoch_loss_values = []
            for data in train_dataloader:
                data = data.to(self.device)
                optimizer.zero_grad()
                loss = self.diffusion.loss(data)
                loss.backward()
                optimizer.step()
                loss_value = loss.cpu().item()
                epoch_loss_values.append(loss_value)
            self.loss_values.append(np.mean(epoch_loss_values))
            if epoch % 10 == 0:
                print(f'{epoch = } Loss = {self.loss_values[-1]}')

            torch.save(self.punet_model.state_dict(), os.path.join(
                PATH_TO_MODELS, f'model_{self.overlap_str}_{self.N}.pth'))
        save_plot_data([self.loss_values], [
                       f'N={self.N},{self.overlap_str} data'], f"loss_{self.overlap_str}_{self.N}.png")

    @torch.inference_mode()
    def generate_new_data(self, sample_count):
        '''
        Generating new samples from the trained model.
        '''
        self.punet_model.eval()
        with torch.no_grad():
            batch_render_data = torch.randn(
                (sample_count, len(self.data_loader.columns), 2, self.N), device=self.device)
            # go from noise to motion data for `noise_steps` steps
            for timestamp in tqdm(reversed(range(1, self.noise_steps)), total=self.noise_steps):
                batch_render_data = self.diffusion.sample_data(
                    batch_render_data, timestamp)

        for index in range(sample_count):
            render_data = batch_render_data[index]

            x_axis_data, y_axis_data = render_data[:,
                                                   0, :], render_data[:, 1, :]
            # unormalizing data back for display.
            render_data[:, 0, :] = ((x_axis_data - torch.min(x_axis_data)) / (torch.max(x_axis_data) - torch.min(
                x_axis_data))) * (self.data_loader.max_x - self.data_loader.min_x) + self.data_loader.min_x
            render_data[:, 1, :] = ((y_axis_data - torch.min(y_axis_data)) / (torch.max(y_axis_data) - torch.min(
                y_axis_data))) * (self.data_loader.max_y - self.data_loader.min_y) + self.data_loader.min_y

            # converting to json for rendering it.
            json_render_data = convert_to_json_format(
                self.data_loader.columns, render_data)
            path_to_file = os.path.join(
                PATH_TO_RESULT_FILES, f'gif{index}_{self.overlap_str}_{self.N}.gif')
            render_seq(json_render_data, path_to_file)
