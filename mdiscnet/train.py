import numpy as np
import os
import torch
from tqdm import tqdm
from lossfn import loss_var
import wandb
# Reference: https://avandekleut.github.io/vae/

def train_model(model, data, config, save_dir):
    
    CHECKPOINT_PATH = './checkpoint.tar'
    model = model.to(model.device)
    
    # Optimiser
    opt = torch.optim.Adam(params=model.parameters(), lr=config.lr)
        
    # initial values
    epoch = 0
    lowest_loss = 1e10
    
    # Start wandb run for logging
    wandb.init(project='mdiscnet', job_type='train', config=config, id='t31z1ck9', resume='must')
    
    if wandb.run.resumed:
        #checkpoint = torch.load(wandb.restore(CHECKPOINT_PATH))
        checkpoint = torch.load(save_dir+CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
    # history
    hist = {'loss':np.zeros(config.epochs),
            'lr':np.zeros(config.epochs)}
    
    # TRAINING CODE
    while epoch < config.epochs:    
        # set into train mode
        model.train()
        opt.param_groups[0]['lr'] = config.lr#*(1-epoch/config.epochs) # decaying lr at each epoch
        
        for x, y in data:
            x = x.to(model.device) # GPU
            opt.zero_grad()
            recon_x, mu, logvar = model(x) # vae model
            loss = loss_var(recon_x, x, mu, logvar, model.beta)[0] # vae loss function
            loss.backward()
            opt.step()
        
        # metrics
        hist['loss'][epoch] = loss.item()
        hist['lr'][epoch] = opt.param_groups[0]['lr']
        
        
        metrics = {
                "train/train_loss": hist['loss'][epoch],
                "train/lr": opt.param_groups[0]['lr'],
                "train/epoch": epoch + 1,
            }
        
        # Log metrics
        wandb.log(metrics)
        
            
        # Check if achieved new lowest loss, if so, save model weights locally and on wandb
        if loss < lowest_loss:
            lowest_loss = hist['loss'][epoch]
            tags = [f"lowest_loss"]
            model_checkpoint(model=model, tags=tags, save_dir=save_dir)

        # Update the progress bar, save model weights locally and on wandb
        if (epoch+1)%100 == 0 or epoch+1 == int(config.epochs):
            tqdm.write(f"Epoch: {epoch+1}/{config.epochs}, Loss: {loss.item():.4f}")
            tags = [f"epoch_{epoch+1}"]
            model_checkpoint(model=model, tags=tags, save_dir=save_dir)
            
        
        # Save checkpoint locally
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': hist['loss'][epoch],
            }, save_dir+CHECKPOINT_PATH)
        
        wandb.save(CHECKPOINT_PATH) # saves checkpoint to wandb
        
        epoch += 1
        
    wandb.finish()
    
    return model, hist

def model_checkpoint(model, tags, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.eval()
    ckpt_file = f"{save_dir}model.pth"
    torch.save(model.state_dict(), ckpt_file)

    artifact_name = f"{wandb.run.id}_model"
    at = wandb.Artifact(artifact_name, type="model")
    at.add_file(ckpt_file)
    wandb.log_artifact(at, aliases=tags)
    print("Checkpoint saved!")