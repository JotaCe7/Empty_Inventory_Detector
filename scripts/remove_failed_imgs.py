#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREAMBLE
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import argparse
import pandas as pd
import os

IMG_FOLDER = '../data/train_test_SKU'
BAD_PATH='../data/bad_data'

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
    # bash run: `python3 scripts/remove_failed_imgs.py "./data/bad_data/failed_imgs.csv" ` -> For failed tagged images
    # bash run: 
    #   `python3 scripts/remove_failed_imgs.py "./data/bad_data/train/corrupted_imgs.csv" ` -> For corrupted train images
    #   `python3 scripts/remove_failed_imgs.py "./data/bad_data/val/corrupted_imgs.csv" `   -> For corrupted val images
    
    failed_imgs = pd.read_csv(failed_path, index_col='img_name')
    img_list = failed_imgs.index
    
    for img_name in img_list:
        
        folder = img_name.split('_')[0]
        img_path = os.path.join(IMG_FOLDER[1:],folder,'images',img_name)
        
        if os.path.exists(img_path):
            print(f'{img_name} removed.')
            os.remove(img_path)
            
if __name__ == "__main__":
    args = parse_args()
    remove_failed_imgs(args.failed_path)                               