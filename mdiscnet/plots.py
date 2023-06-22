import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

def plot_latent(model, data):
    N = len(data)
    kh_arr = np.zeros(N)
    rmp_arr = np.zeros(N)
    z_arr = np.zeros((N,2))
    
    fig, ax = plt.subplots(1,3,sharex=False, sharey=False, figsize=(10,5))
    
    for i, (x, y) in enumerate(data):
        z,*_ = model.encode(x.to(model.device))
        z = z.to('cpu').detach().numpy()
        z_arr[i] = z
        
        # split label into rmp and kh colorbars
        label = y[0].split("_")
        kh = float(label[0][2:])
        rmp = float(label[1][3:])
        kh_arr[i] = kh
        rmp_arr[i] = rmp
        
    s1 = ax[0].scatter(z_arr[:, 0], z_arr[:, 1], c=kh_arr, cmap='jet')
    s2 = ax[1].scatter(z_arr[:, 0], z_arr[:, 1], c=rmp_arr, cmap='jet')
    s3 = ax[2].scatter(rmp_arr, kh_arr)
    ax[2].set_xlabel('rmp')
    ax[2].set_ylabel('Kh')
    plt.colorbar(s1, ax=ax[0], label='kh')
    plt.colorbar(s2, ax=ax[1], label='rmp')
    plt.tight_layout()
    
def reconstruct(model, x, width):
    z, *_ = model.encode(x)
    x_hat = model.decode(z)
    x_hat = x_hat.reshape(width, width).to('cpu').detach().numpy()
    return x_hat
    
def decode_and_plot(z):
    x_hat = model.decode(z)
    x_hat = x_hat.reshape(width, width).to('cpu').detach().numpy()
    plt.imshow(x_hat)
    return x_hat
    
def plot_reconstructed(model, width, z0=(-5, 5), z1=(-5, 5), n_img=12):
    img = np.zeros((n_img*width, n_img*width)) # n*width rows by n*width columns
    # loop through values in the latent space
    for i, y in enumerate(np.linspace(*z1, n_img)):
        for j, x in enumerate(np.linspace(*z0, n_img)):
            z = torch.Tensor([[x, y]]).to(model.device)
            x_hat = model.decode(z)
            x_hat = x_hat.reshape(width, width).to('cpu').detach().numpy()
            img[(n_img-1-i)*width:(n_img-1-i+1)*width, j*width:(j+1)*width] = x_hat
    plt.imshow(img, extent=[*z0, *z1], cmap='jet')

    
def interpolate(model, x_1, x_2, width, n_img=12):
    z_1, *_ = model.encode(x_1)
    z_2, *_ = model.encode(x_2)

    # interpolate between z_1 and z_2 
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n_img)])
    interpolate_list = model.decode(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    img = np.zeros((width, n_img*width))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i*width:(i+1)*width] = x_hat.reshape(width, width)
    plt.imshow(img, cmap='jet')
    plt.xticks([])
    plt.yticks([])
    
def interpolate_gif(model, width, filename, x_1, x_2, n_img=100):
    z_1, *_ = model.encode(x_1)
    z_2, *_ = model.encode(x_2)

    # interpolate between z_1 and z_2 
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n_img)])
    interpolate_list = model.decode(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()*255

    images_list = [Image.fromarray(img.reshape(width, width)).resize((256, 256)) for img in interpolate_list]
    images_list = images_list + images_list[::-1] # loop back beginning

    images_list[0].save(
        f'{filename}.gif',
        save_all=True,
        append_images=images_list[1:],
        loop=1)