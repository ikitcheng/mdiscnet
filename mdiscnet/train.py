import numpy as np
import torch
from tqdm import tqdm
from lossfn import loss_var
import wandb
# Reference: https://avandekleut.github.io/vae/

def train_model(model, data, config):
    
    model = model.to(model.device)
    
    # Start wandb run for logging
    wandb.init (project='mdiscnet', job_type='train', config=config)

    opt = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    
    hist = {'loss':np.zeros(config.epochs)}

    for epoch in tqdm(range(config.epochs)):
        for x, y in data:
            x = x.to(model.device) # GPU
            opt.zero_grad()
            recon_x, mu, logvar = model(x) # vae model
            loss = loss_var(recon_x, x, mu, logvar, model.beta)[0] # vae loss function
            loss.backward()
            opt.step()
            
        hist['loss'][epoch] = loss.item()
        
        metrics = {
                "train/train_loss": hist['loss'][epoch],
                "train/epoch": epoch + 1,
            }
        
        # Log metrics over time to visualize performance
        wandb.log(metrics)

        # Update the progress bar with relevant information
        if epoch%100 == 0:
            tqdm.write(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
    wandb.finish()
    
    return model, hist