#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREAMBLE
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Python imports
import os 
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Self-made
from utils_data import util_funcs 
from utils_data import cons

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
        


def get_bboxes(img_path: str = os.path.join(cons.IMG_FOLDER,'train/images/train_0.jpg')) -> pd.DataFrame:
    """ 
    It gets the coordinates of the bounding boxes corresponding to
    the image passsed in `img_path`.

    Parameters
    ----------
    img_path: str
        Path to image.
    ----------
    box_coordinates: pd.DataFrame
        DataFrame with coordinates of the bounding boxes.
    """
 
    # Read the image
    img_name = os.path.split(img_path)[-1]
    print(img_name)
    box_coordinates = get_bbox_coords(img_name)
    
    return box_coordinates
    
def plot_bboxes(img_path: str = os.path.join(cons.IMG_FOLDER,'train/images/train_0.jpg') , box_coordinates: pd.DataFrame = pd.DataFrame(),axes: Axes = None, skip_plot: bool = False):
    """ 
    It plots the bounding boxes in green when products are present.
    If there are missing products, then red bboxes are drawn.

    Parameters
    ----------
    img_path: str
        Path to image.
        
    box_coordinates: pd.DataFrame
        Contains the image coordinates to plot. 
        Default = None: searches for the coordinates in the static dataset (.csv)
        stored under `data/SKU110K/annotations/annotations.csv`.
        
    axes: matplotlib.axes.Axes (Optional)
        Axes in which to plot the image.
        
    skip_plot: bool
        Wether to skip or not the plot of the image.
        
    Returns
    ----------
    img: np.array (Optional)
        Image plotted.
    """
    #Read the image
    img = cv2.imread(img_path)
    # Get BBox coordinates
    if box_coordinates.empty:
        box_coordinates = get_bboxes(img_path=img_path)
    
    # Plot all boxes
    for _, box_coords in box_coordinates.iterrows():
        
        x1, y1, x2, y2 = (box_coords.x1, box_coords.y1, box_coords.x2, box_coords.y2)
        
        if box_coords['class'] == 1:
            img = cv2.rectangle(img, (x1, y1), (x2, y2), cons.GREEN, thickness=5)
        else:
            img = cv2.rectangle(img, (x1, y1), (x2, y2), cons.RED, thickness=5)
            
    # Plot image with boxes
    if not skip_plot:
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