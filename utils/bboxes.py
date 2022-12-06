#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREAMBLE
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import os 
import cv2
import numpy as np
import pandas as pd
from utils import util_funcs 

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


IMG_PATH = '../data/train_test_SKU/images'
ANNOT_PATH = '../data/SKU110K/annotations'


BLUE =  (255, 0, 0)  
GREEN = (0, 255, 0)  
RED =   (0, 0, 255)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_bbox_coords(img_name: str='train_0.jpg') -> pd.DataFrame:
    """ 
    Gets the box coordinates of given image in a dataframe format.

    Parameters
    ----------
    img_name: str
        The image name.
    Returns
    ----------
    box_coords: pd.DataFrame 
        Dataframe with the box coordinates for given image.
    """
    try: 
        
        img_set = img_name.split('_')[0]
        # Search the image in the csv in chunks
        for chunk_df in util_funcs.read_csv_chunks(img_set):
            
            img_df = chunk_df[ chunk_df.img_name == img_name ]
            if not img_df.empty: break
        
        # Get the coordinates        
        box_coords = img_df[[ 'x1', 'y1', 'x2', 'y2']] 
        
        return box_coords
    
    except NameError:
        print('Image name does not exist in the dataset' )
        


def get_bboxes(img_name: str='train_0.jpg', axes: Axes = None, plot_f: bool = True ) -> np.array:
    """ 
    It plots the bounding boxes in green.

    Parameters
    ----------
    img_name: str
        The image name.
    axes: matplotlib.axes.Axes (Optional)
        Axes in which to plot the image.
    plot_f: booL
        Wether to plot or not the image.
    Returns
    ----------
    img: np.array (Optional)
        Image plotted.
    """
    # Build path to image
    ttv = img_name.split('_')[0]
    img_path = os.path.join(IMG_PATH, ttv, img_name)
    
    # Read the image
    img = cv2.imread(img_path)
    box_coordinates = get_bbox_coords(img_name)
    
    # Plot all boxes
    for _, box_coords in box_coordinates.iterrows():
        x1, y1, x2, y2 = (box_coords.x1, box_coords.y1, box_coords.x2, box_coords.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, thickness=5)

    # Plot image with boxes
    if plot_f:
        if axes:
            axes.imshow(img)
        else:
            plt.imshow(img)   
    
    return img


def get_bbox_area(box_coords: tuple, plot_f: bool = True):
    """ 
    It computes the bounding boxes compound area and the image total area.

    Parameters
    ----------
    box_coords: tuple
        Bounding box coordinates (x1,y1,x2,y2)
    plot_f: booL
        Wether to plot or not the image.
    Returns
    ----------
    areas: tuple
        Bounding box area and total area
    """

    x1, y1, x2, y2 = box_coords
    bb_area = (y1-y2)*(x1-x2)
    
    return bb_area


def get_bboxes_total_area(tags_df: pd.DataFrame):
    """ 
    Get the total area of the bounding boxes and the 
    percentage in relation to the total image area.

    Parameters
    ----------
    tag_df: pd.DataFrame
        The dataframe containing the images and it's tags. 
        
    Returns
    ----------
    aread_df: pd.DataFrame
        Dataframe with area related information.
    """    
    # Calculate total_area of the images
    tags_df['total_area'] = tags_df.total_height * tags_df.total_width

    # Calculate area of bboxes in the images
    tags_df['bbox_area'] = tags_df.apply(lambda r: get_bbox_area( (r.x1,r.y1,r.x2,r.y2) ), axis = 1)

    # See percentage of area covered by bboxes 
    tags_df['bbox_area_perc']  = tags_df.bbox_area / tags_df.total_area

    # Get only areas related columns
    areas_df = tags_df[ ['total_area','bbox_area','bbox_area_perc'] ]
    # areas_df.set_index('img_name', inplace= True)
    
    return areas_df