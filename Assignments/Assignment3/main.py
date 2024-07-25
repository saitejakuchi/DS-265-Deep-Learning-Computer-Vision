import os
import yaml
import torch
import random
import numpy as np
from utils import PATH_TO_CONFIG_FILE, PATH_TO_RESULT_FILES, MotionDataLoader, TrainingLoop, save_plot_data


def set_seed(seed: int = 41) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def train_model(configuration_data):
    loader = MotionDataLoader()
    train_obj = TrainingLoop(loader, configuration_data)
    if configuration_data['is_train']:
        train_obj.train()
    train_obj.generate_new_data(configuration_data['sample_count'])
    return train_obj.loss_values


def main():
    set_seed()
    os.makedirs(PATH_TO_RESULT_FILES, exist_ok=True)
    with open(PATH_TO_CONFIG_FILE) as f:
        configurations = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configurations['device'] = device
    sampling_values = [10, 16, 24, 32, 48, 60]
    loss_array_data = []
    label_data = []
    for overlap_value in [True, False]:
        if overlap_value:
            overlap_str = 'overlap'
        else:
            overlap_str = 'nonoverlap'
            configurations['batch_size'] = 1
        for N in sampling_values:
            configurations['overlap'] = overlap_value
            configurations['N'] = N
            loss_data = train_model(configurations)
            loss_array_data.append(loss_data)
            label_data.append(f'N={N},{overlap_str} data')
    if configurations['is_train']:
        save_plot_data(loss_array_data, label_data,
                       configurations['path_to_plot_file'])


if __name__ == '__main__':
    main()
