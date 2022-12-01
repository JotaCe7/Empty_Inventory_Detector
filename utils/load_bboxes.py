#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREAMBLE
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import os, cv2, pandas as pd
import matplotlib.pyplot as plt
import numpy as np

IMG_PATH = '../data/train_test_SKU'
ANNOT_PATH = '../data/SKU110K/annotations'

BLUE =  (255, 0, 0)  
GREEN = (0, 255, 0)  
RED =   (0, 0, 255)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def read_img_boxes(img_name: str='train_0.jpg', chunksize: int=10000) -> pd.io.parsers.TextFileReader:
    """ 
    Creates a dataframe chunk to iterate over, in order to read the bounding 
    boxes coordinates in the annotation csv files.

    Parameters
    ----------
    img_name: str
        The image name.
    chunksize: int
        Size of the Dataframe chunk.
    Returns
    ----------
    chunk: TextFileReader 
        Iterator to read Dataframe in chunks.
    """
    # Build path to annotation file
    ttv = img_name.split('_')[0]
    annot_file = 'annotations_' + ttv + '.csv'
    annotation_path = os.path.join(ANNOT_PATH, annot_file)
    
    return pd.read_csv(annotation_path, names=['img_name', 'x1', 'y1', 'x2', 'y2', 'type', 'height', 'width'], chunksize=chunksize)


def get_boxes(img_name: str='train_0.jpg') -> pd.DataFrame:
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
        # Search the image in the csv in chunks
        for chunk_df in read_img_boxes(img_name):
            
            img_df = chunk_df[ chunk_df.img_name == img_name ]
            if not img_df.empty: break
        
        # Get the coordinates        
        box_coords = img_df[[ 'x1', 'y1', 'x2', 'y2']] 
        
        return box_coords
    
    except NameError:
        print('Image name doesn not exist in the dataset' )
        


def plot_bounding_boxes(img_name: str='train_0.jpg') -> np.array:
    """ 
    It plots the bounding boxes in green.

    Parameters
    ----------
    img_name: str
        The image name.
    train_test_val: str
        Defines the set (train, test or val) to read. .
    Returns
    ----------
        None
    """
    # Build path to image
    ttv = img_name.split('_')[0]
    img_path = os.path.join(IMG_PATH, ttv, img_name)
    
    # Read the image
    img = cv2.imread(img_path)
    box_coordinates = get_boxes(img_name)
    
    # Plot all boxes
    for _, box_coords in box_coordinates.iterrows():
        x1, y1, x2, y2 = (box_coords.x1, box_coords.y1, box_coords.x2, box_coords.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, thickness=5)

    # Plot image with boxes
    plt.imshow(img)
    
    return img