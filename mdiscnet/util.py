import torch
import pickle 
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

def save_result(result, filename):
    """ Save pickle data. """
    with open(filename, "wb") as f:
        pickle.dump(result, f)

def load_result(filename):
    """ Load pickle data """
    with open(filename, "rb") as f:
        result = pickle.load(f)
    return result