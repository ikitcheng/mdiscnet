import torch

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    
def load_model(model, model_path):
    return model.load_state_dict(torch.load(model_path))