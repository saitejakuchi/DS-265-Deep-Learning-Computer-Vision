from utils import *


def main():
    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    batch_size = 1
    num_classes = 0
    num_of_steps = 1000
    cin = 2
    cout = 2
    noise_value = 0
    diffusion_model = Diffusion(num_of_steps, path_to_config=None, noise_level=noise_value,
                                c_in=cin, c_out=cout, num_classes=num_classes, device=device, batch_size=batch_size)
    diffusion_model.load_weights('./models/SongNet_no_finetune_46.pt')
    diffusion_model.train(epochs)


if __name__ == '__main__':
    main()
