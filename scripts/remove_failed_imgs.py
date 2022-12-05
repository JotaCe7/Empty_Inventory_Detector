#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREAMBLE
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import argparse
import pandas as pd
import os

IMG_PATH = '../data/train_test_SKU'

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Python Arg Parser
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Remove failed images")
    parser.add_argument(
        "failed_path",
        type=str,
        help=(
            "Full path to the directory where the list of failed imaes is. \
             E.g. `/home/app/src/data/train_test_SKU/failed_imgs.csv`."
        ),
    )

    args = parser.parse_args()

    return args

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Script: Remove failed and corrupted images
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def remove_failed_imgs(failed_path: str):
    """ 
    Function to delete the hardlinked images on `img_list'
    Parameters
    ----------
    failed_path: str 
        Path to the list of image names to delete from the dataset.
    Returns
    ----------
        None
    """
    failed_imgs = pd.read_csv(failed_path)
    img_list = failed_imgs.index
    
    for img_name in img_list:
        
        folder = img_name.split('_')[0]
        img_path = os.path.join(IMG_PATH,folder,img_name)
        
        if os.path.exists(img_path):
            print(f'{img_name} removed.')
            os.unlink(img_path)
            
if __name__ == "__main__":
    args = parse_args()
    remove_failed_imgs(args.failed_path)                               