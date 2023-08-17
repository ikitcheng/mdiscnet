from glob import glob
import os
import time
from pymagdisc.data.load_data import load_model
from pymagdisc import config
import load_catalog
import util

def read_filenames(filenames_path:str):
    """ Read from text file containing the model filenames.
    """
    with open(filenames_path) as f:
        filenames = f.read().splitlines()
    return filenames

def create_model_dict(kh, rmp, nr, nmu, c2d_r, c2d_mu, v2d_alpha, v2d_Br=None, v2d_Bth=None, v2d_PcEq=None, v2d_PhEq=None):
    return {'kh':kh, 
            'rmp':rmp,
            'nr':nr,
            'nmu':nmu,
            'c2d':{'r':c2d_r, 
                   'mu':c2d_mu},
            'v2d':{'alpha':v2d_alpha, 
                   'Br':v2d_Br,
                   'Bth':v2d_Bth,
                   'PcEq':v2d_PcEq,
                   'PhEq':v2d_PhEq}
           }

def download_data_from_server(filename):
    # Check if already file already exists:
    if not os.path.isfile(f"./data/MDISC/small_dataset/{filename}"):
        print("File does not exist. Downloading...")
        # Get file using SSH and SCP
        remote_host = "ikc19@zuserver1.star.ucl.ac.uk"
        expect_str = "zuserver1%"
        child = load_catalog.ssh_to(remote_host, expect_str, verbose=False)
        origin = f"ikc19@zudata6.star.ucl.ac.uk:/export/zudata6/alex/Jupiter_Catalogs/{filename}"
        destination = f"ikc19@hypatia-login.hpc.phys.ucl.ac.uk:{config.PATH_TO_DATA}{filename}"
        load_catalog.send_files(child, origin, destination, verbose=False)
        child.close()
    else:
        print(f"{filename} exists already.")

def extract_data(filename):
    filename_split = filename.split('_')
    kh, rmp, nr, nmu = str(filename_split[2][2:]), str(filename_split[3][3:]), str(601), str(601)

    # Load .mat file 
    MD = load_model(f"{config.PATH_TO_DATA}{filename}")
    
    # extract data
    c2d_r = MD["c2d"]["r"]
    c2d_mu = MD["c2d"]["mu"]
    v2d_alpha = MD["v2d"]["alpha"]
    v2d_Br = None #MD["v2d"]["Br"]
    v2d_Bth = None #MD["v2d"]["Bth"]
    v2d_PcEq = None #MD["v2d"]["PcEq"]
    v2d_PhEq = None #MD["v2d"]["PhEq"]
    
    # store in dict
    model_dict = create_model_dict(kh, rmp, nr, nmu, c2d_r, c2d_mu, v2d_alpha, v2d_Br, v2d_Bth, v2d_PcEq, v2d_PhEq)
    return model_dict