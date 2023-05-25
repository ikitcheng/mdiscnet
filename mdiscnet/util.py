import torch

def save_model(model, model_path):
    """ Save the model weights. """
    torch.save(model.state_dict(), model_path)
    
def load_model(model, model_path):
    """ Load the model with saved weights on 
    the same device it was trained and saved on. """
    model.load_state_dict(torch.load(model_path))
    model.to(model.device)
    model.eval()
    return model