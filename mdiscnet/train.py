import numpy as np
import torch
from tqdm import tqdm
from lossfn import loss_var
# Reference: https://avandekleut.github.io/vae/

def train_model(model, data, lr=1e-3, epochs=20):
    model = model.to(model.device)
    opt = torch.optim.Adam(params=model.parameters(), lr=lr)
    hist = {'loss':np.zeros(epochs)}
    for epoch in tqdm(range(epochs)):
        for x, y in data:
            x = x.to(model.device) # GPU
            opt.zero_grad()
            recon_x, mu, logvar = model(x) # vae model
            loss = loss_var(recon_x, x, mu, logvar, model.beta)[0] # vae loss function
            loss.backward()
            opt.step()
            
        hist['loss'][epoch] = loss.item()
        # Update the progress bar with relevant information
        if epoch%100 == 0:
            tqdm.write(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    return model, hist