import torch
from torch.nn import functional as F

def loss_var(recon_x, x, mu, logvar, beta):
    """ Loss function for vae """
    # https://stats.stackexchange.com/questions/350211/loss-function-autoencoder-vs-variational-autoencoder-or-mse-loss-vs-binary-cross

    # https://github.com/pytorch/examples/issues/399

    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.shape[0] # pixel 방향으로 sum, minibatch 방향으로 mean

    KLD_temp = 1 + logvar - mu.pow(2) - logvar.exp() # shape: minibatch x latent

    KLD = -0.5 * torch.mean(torch.sum(KLD_temp,axis=1)) # latent 방향으로 sum, minibatch 방향으로 mean


    # recon_x_ = recon_x * torch.tensor(ff_max).to(device)[coef_idx] + torch.tensor(ff_min).to(device)[coef_idx]

    # x_ = x * torch.tensor(ff_max).to(device)[coef_idx] + torch.tensor(ff_min).to(device)[coef_idx]

    # MSE = F.mse_loss(recon_x, x, reduction='mean') # VAEROM 논문(PoF)도 Mse의 mean을 recon error로 사용
    
    MSE = F.mse_loss(recon_x, x, reduction='sum') / x.shape[0] # VAEROM 논문(PoF)도 Mse의 mean을 recon error로 사용
    

    return (MSE + beta * KLD), KLD, (beta * KLD), MSE


      

def loss_novar(recon_x, x, beta, coef_idx = 0):
    """ Loss function for AE """
    # https://stats.stackexchange.com/questions/350211/loss-function-autoencoder-vs-variational-autoencoder-or-mse-loss-vs-binary-cross

    # https://github.com/pytorch/examples/issues/399

    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.shape[0] # pixel 방향으로 sum, minibatch 방향으로 mean

    

    recon_x_ = recon_x * torch.tensor(ff_max).to(device)[coef_idx] + torch.tensor(ff_min).to(device)[coef_idx]

    x_ = x * torch.tensor(ff_max).to(device)[coef_idx] + torch.tensor(ff_min).to(device)[coef_idx]

    MSE = F.mse_loss(recon_x, x, reduction='mean') # VAEROM 논문(PoF)도 Mse의 mean을 recon error로 사용

    MSE_loss_fn_scale = F.mse_loss(recon_x, x, reduction='sum') / x.shape[0]

    MSE_realscale = F.mse_loss(recon_x_, x_, reduction='mean')

    return MSE, MSE_realscale, MSE_loss_fn_scale