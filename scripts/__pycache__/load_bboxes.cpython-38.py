# uncompyle6 version 3.8.0
# Python bytecode 3.8.0 (3413)
# Decompiled from: Python 3.8.10 (default, Jun 22 2022, 20:18:18) 
# [GCC 9.4.0]
# Embedded file name: /mnt/d/User/NicoSan/NicoSan/Personales/AICarreer/AIBootcamp/Repos/FinalProject/notebooks/../scripts/load_bboxes.py
# Compiled at: 2022-12-01 16:15:57
# Size of source mod 2**32: 3900 bytes
import os, cv2, pandas as pd
import matplotlib.pyplot as plt
import numpy as np
IMG_PATH = '../data/train_test_SKU'
ANNOT_PATH = '../data/SKU110K/annotations'

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
    ttv = img_name.split('_')[0]
    annot_file = 'annotations_' + ttv + '.csv'
    annotation_path = os.path.join(ANNOT_PATH, annot_file)
    return pd.read_csv(annotation_path, names=['img_name', 'x1', 'y1', 'x2', 'y2', 'type', 'height', 'width'], chunksize=chunksize)


def get_boxes(img_name: str='train_0.jpg')


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
    ttv = img_name.split('_')[0]
    img_path = os.path.join(IMG_PATH, ttv, img_name)
    print(img_path)
    img = cv2.imread(img_path)
    box_coordinates = get_boxes(img_name)
    for _, box_coords in box_coordinates.iterrows():
        x1, y1, x2, y2 = (box_coords.x1, box_coords.y1, box_coords.x2, box_coords.y2)
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=5)
    else:
        plt.imshow(img)
        return img