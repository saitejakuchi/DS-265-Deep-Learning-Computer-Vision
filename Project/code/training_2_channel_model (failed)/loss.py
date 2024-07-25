import torch
import numpy as np

class SureLoss:
    def __init__(self, sigma_w = 0, sigma_min=0.02, sigma_max=100, eps = 0.001) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_w_value = sigma_w
        self.eps = eps
        self.lamda_value = None

    def __call__(self, net, images, labels=None, augment_pipe=None, **kwargs):
        '''
        train
        '''
        # import pdb;pdb.set_trace()
        y = images
        x = labels
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        self.sigma_w = torch.rand([images.shape[0], 1, 1, 1], device=images.device) * self.sigma_w_value
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        x_tilda = y # with sigma_w noise.
        tweedie_residual = torch.multiply(self.sigma_w ** 2, net(x_tilda, self.sigma_w[:,0,0,0], labels, augment_labels=augment_labels))
        tweedie_reparam = x_tilda + tweedie_residual   

        perturbation = torch.randn_like(x_tilda) * self.sigma_w * self.eps
        x_perturbed = x_tilda + perturbation
        self.sigma_w_pertubed = self.sigma_w + self.eps * self.sigma_w
        tweedie_residual_perturbed = torch.multiply(self.sigma_w_pertubed ** 2, net(x_perturbed, self.sigma_w_pertubed[:,0,0,0], labels, augment_labels=augment_labels))
        tweedie_reparam_perturbed = x_perturbed + tweedie_residual_perturbed
        meas_loss = net(tweedie_reparam + n, sigma[:,0,0,0], labels, augment_labels=augment_labels) + n / sigma ** 2
        dsm_loss = torch.linalg.norm(meas_loss) * sigma ** 2

        sure1_loss = torch.linalg.norm(tweedie_residual)

        div_loss = 2 * (self.sigma_w ** 2) * torch.mean(perturbation * (tweedie_reparam_perturbed - tweedie_reparam) / self.eps, dim=(-1, -2, -3))

        sure_loss = sure1_loss + div_loss
        
        if self.sigma_w_value == 0:    
            self.lamda_value = 0.0001 if self.lamda_value is None else self.lamda_value
        else:
            self.lamda_value = dsm_loss/sure_loss if self.lamda_value is None else self.lamda_value
            
        sure_score_loss = dsm_loss + self.lamda_value * sure_loss
        # print(f'{self.lamda_value=}')
        return sure_score_loss

