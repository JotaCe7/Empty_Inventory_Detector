
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREAMBLE
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as  np
from utils import bboxes

IMG_PATH = '../data/train_test_SKU'
ANNOT_PATH = '../data/SKU110K/annotations'

CRITERIA = ['area','n_bboxes']

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def read_csv_chunks(img_set: str='train', chunksize: int=10000) -> pd.io.parsers.TextFileReader:
    """ 
    Creates a dataframe chunk to iterate over, in order to read the bounding 
    boxes coordinates in the annotation csv files.

    Parameters
    ----------
    img_set: str
        The image set (train,test or val).
    chunksize: int
        Size of the Dataframe chunk.
    Returns
    ----------
    chunk: TextFileReader 
        Iterator to read Dataframe in chunks.
    """
    # Build path to annotation file
    ttv = img_set.split('_')[0]
    annot_file = 'annotations_' + ttv + '.csv'
    annotation_path = os.path.join(ANNOT_PATH, annot_file)
    
    return pd.read_csv(annotation_path, names=['img_name', 'x1', 'y1', 'x2', 'y2', 'type', 'height', 'width'], chunksize=chunksize)


def get_failed_imgs(tags_df: pd.DataFrame, criterion: str = 'area', thresh: float = 0.01, verbose: bool = True) -> list:
    """ 
    Get a list of failed images (based on criterion)

    Parameters
    ----------
    tag_df: pd.DataFrame
        The dataframe containing the images and it's tags. 
    criterion: str
        The criterion to use to decide for a failed images.
        It can be either 'area' or 'number of bboxes'
    thresh: float
        The threshold to use fo filter for failed images. 
        If criterion == 'area': this corresponds to the lower quantile.
        If criterion == 'n_bboxes': this corresponds to the min number of bounding boxes.
        
    Returns
    ----------
    failed_imgs: list
        List of failed image names.
    """
    # Check for valid criterion
    if criterion not in CRITERIA:
        raise ValueError("Criterion used not valid. Enter either 'area' or 'n_bboxes'.")
    
    
    if criterion == 'n_bboxes':
        
        # Count images with less that `thresh` tags
        n_tags_per_image = tags_df.groupby('img_name').size().sort_values()
        failed_imgs = n_tags_per_image[n_tags_per_image < thresh]
      
        # Prints information
        if verbose: 
            n_failed = failed_imgs.shape[0]
            print('Number of failed images: ', n_failed)
            print(f'List of failed images:\n{failed_imgs}')
            print(f'\nThis represents {n_failed/len(tags_df)*100}% of the images')

    elif criterion == 'area': 
        
        # Getting area realted dataframe and bbox_arad cover
        areas_df = bboxes.get_bboxes_total_area(tags_df)
        bbox_area_cover = areas_df.groupby('img_name').bbox_area_perc.sum().sort_values()
        
        FAIL_THRESH = np.quantile(bbox_area_cover,q = thresh)
        failed_imgs = bbox_area_cover[bbox_area_cover < FAIL_THRESH]
        
        # Prints information
        if verbose: 
            print('Treshold used:', FAIL_THRESH)
            print('# failed images: ', len(failed_imgs))

    return failed_imgs

def remove_failed_imgs_from_data(img_list: list):
    """ 
    Function to delete the hardlinked images on `img_list'
    Parameters
    ----------
    img_list: list  
        The list of image names to delete from the dataset 
    Returns
    ----------
        None
    """
    
    for img_name in img_list:
        
        folder = img_name.split('_')[0]
        img_path = os.path.join(IMG_PATH,folder,img_name)
        
        if os.path.exists(img_path):
            print(f'{img_name} removed.')
            os.unlink(img_path)