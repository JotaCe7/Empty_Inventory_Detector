
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREAMBLE
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as  np
from PIL import Image
from utils import bboxes

IMG_PATH = '../data/train_test_SKU/images'
ANNOT_PATH = '../data/SKU110K/annotations'
LABEL_PATH = '../data/train_test_SKU/labels'

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
    
    return pd.read_csv(annotation_path, names=['img_name', 'x1', 'y1', 'x2', 'y2', 'type', 'total_height', 'total_width'], chunksize=chunksize)

def drop_missing_img(imgs: pd.Series) -> pd.Series:
    """ 
    Get a list of failed images (based on criterion)

    Parameters
    ----------
    imgs: pd.DataFrame
        The dataframe containing the images and it's tags. 

    Returns
    ----------
    clean_imgs: pd.Series
        Series of existent images.
    """

    # Check if the img_name exist. Drop it if it doesn't 
    for img in imgs.index:
            # Build path to image
        ttv = img.split('_')[0]
        img_path = os.path.join(IMG_PATH, ttv, img) 

        if not os.path.exists(img_path): 
            imgs.drop(img, inplace= True)
    
    
    return imgs

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
    verbose: bool
        Wether extra information will be printed or not.
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
      
        # Check if the img_name exist. Drop it if it doesn't 
        failed_imgs = drop_missing_img(failed_imgs)
      
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
        
        # Check if the img_name exist. Drop it if it doesn't 
        failed_imgs = drop_missing_img(failed_imgs)
        
        # Prints information
        if verbose: 
            print('Treshold used:', FAIL_THRESH)
            print('# failed images: ', len(failed_imgs))

    return failed_imgs            
            
def detected_corrupted_imgs(tags_df: pd.DataFrame) -> list:
    """ 
    Detects corrupted images in the dataset
    ----------
    tag_df: pd.DataFrame
        The dataframe containing the images and it's tags. 
    Returns
    ----------
    corrupted_imgs: list
        List of corrupted images sorted by name. 
    """
        
    img_list = set(tags_df.index)

    corrupted_imgs = []
    for img_name in img_list:
        
        # Build path to img
        folder = img_name.split('_')[0]
        img_path = os.path.join(IMG_PATH,folder,img_name)
        # Read img
        try: 
            img = Image.open(img_path)
            img.getdata()[0]
        except:
            # Append the corrupted img_name and continue
            corrupted_imgs.append(img_name)
            continue
       
    return sorted(corrupted_imgs)
    
def to_yolov5_coords(original_tags_df:pd.DataFrame) -> pd.DataFrame:
    """ 
    Converts the bboxes coordinates from format `xmin, ymin, xmax, ymax`
    to center_x, center_y, width height`  
    ----------
    original_label_df: pd.DataFrame
        Dataframe containing the image names and it's tags using
        `xmin, ymin, xmax, ymax`coordinates 
    Returns
    ----------
    yolo_labels_df: pd.DataFrame
        Dataframe with the bboxes coordinates converted to yolo
        format: `class_id,center_x, center_y, width height`  
    """
    
    n_samples = len(original_tags_df)    
    
    # Get original coordinates
    xmin_coords = original_tags_df.x1 
    ymin_coords =  original_tags_df.y1
    xmax_coords =  original_tags_df.x2
    ymax_coords = original_tags_df.y2
    Width = original_tags_df.total_width
    Heigth = original_tags_df.total_height
    
    
    # Compute YOLO coordinates
    width_bb = (xmax_coords.values - xmin_coords.values) / Width
    height_bb = (ymax_coords.values - ymin_coords.values) / Heigth
    center_x = width_bb / 2 / Width
    center_y = height_bb / 2 / Heigth

    # Create the new Dataframe with the corresponding columns
    yolo_labels_df = pd.DataFrame()
    
    yolo_labels_df.index = original_tags_df.index
    yolo_labels_df['class_id'] = 0
    yolo_labels_df['center_x'] = center_x
    yolo_labels_df['center_y'] = center_y
    yolo_labels_df['width_bb'] = width_bb
    yolo_labels_df['height_bb'] = height_bb
    
    return yolo_labels_df


def labels_to_txt(yolo_labels_df: pd.DataFrame):
    
    img_set = sorted(set(yolo_labels_df.index))
    formatter = ['%d'] + ['%1.8f']*4
    
    os.makedirs(LABEL_PATH, exist_ok= True)
    for img in img_set:
        
        # Filepath
        filename = img.split('.')[0] + '.txt'
        filepath = os.path.join(LABEL_PATH, filename)
        
        # Get coordinate values
        values = np.array(yolo_labels_df.loc[img].values)
        if values.ndim == 1: values = values.reshape(1,-1)
                 
        # Save to text file
        np.savetxt(filepath,values, fmt= formatter)
        if os.path.exists(filepath): print(f' {filename} saved')
    
    