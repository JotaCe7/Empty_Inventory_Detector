
#-----------------------------------------------------------------------------------------------------------------------------
# PREAMBLE
#-----------------------------------------------------------------------------------------------------------------------------
import json
import os
import time
import sys

import numpy as np
import cv2
import redis
import settings
from get_model import get_model

sys.path.append("..")
from utils_data.bboxes import plot_bboxes

#-----------------------------------------------------------------------------------------------------------------------------
# INITIALIZATIONS
#-----------------------------------------------------------------------------------------------------------------------------

# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
                host = settings.REDIS_IP ,
                port = settings.REDIS_PORT,
                db = settings.REDIS_DB_ID
                )

# Load your ML model and assign to variable `model`
model = get_model()


#-----------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------------

    
#Non Max Supression: best bounding box
def NMS(img_tuple, boxes, overlapThresh = 0.4):
    
    """
    Receives `boxes` as a `numpy.ndarray` and gets the best bounding 
    box when there is overlapping bounding boxes.

    Parameters
    ----------
    boxes : numpy.ndarray
        Array with all the bounding boxes in the image.

    Returns
    -------
    best_bboxes: pd.DataFrame
        Dataframe with only the best bounding boxes, 
        in the format: ["xmin","ymin","xmax","ymax","class"]
    """
    
    #return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # compute the area of the bounding boxes and sort the bounding
    
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We have a least a box of one pixel, therefore the +1
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):

        
        temp_indices = indices[indices!=i]
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        
        if np.any(overlap) > overlapThresh:
            
            if box[4] == 0.0:
                continue                            #[ADDED]: Never delete missing boxxes

            indices = indices[indices != i]
            
    best_bboxes =   boxes[indices].astype(int)
    
    img_name = img_tuple[0]
    img_size = img_tuple[1]
    
    best_bboxes_df = pd.DataFrame(data = best_bboxes, index= [img_name]*len(best_bboxes), columns=["x1","y1","x2","y2","class"])
    best_bboxes_df['total_height'] = img_size[0]
    best_bboxes_df['total_width'] = img_size[1]
    
    
    return best_bboxes_df
        
    
    # Creating image with bboxes -> store in '/response/'
def predict_bboxes(img_name):
    """
    Loads the original image and logs the new image
    with the bounding boxes. It stores it a new folder
    called response. 

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    
    # Load image
    img_path = os.path.join(settings.UPLOAD_FOLDER,img_name)
    img = cv2.imread(img_path)
    
    # Get bounding boxes
    output = model(img)
    df = output.pandas().xyxy[0]
    df = df.sort_values("class")
    bboxes = df[["xmin","ymin","xmax","ymax","class"]].to_numpy()
    
    # Non-Max Supression: Filter only best bounding boxes
    best_bboxes = NMS((img_name,img.shape),bboxes, overlapThresh= settings.OVERLAP_THRESH)
    
    # Predict (draw all bounding boxes) and store
    img = plot_bboxes(img_path, box_coordinates= best_bboxes, skip_plot = True )
    pred_img_path = os.path.join(settings.PREDICTIONS_FOLDER, img_name)                                 # store as: "predictions/<img_name>"
    cv2.imwrite(pred_img_path, img)
    

def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        
        # 1. Read the job from Redis
        _ , msg= db.brpop(settings.REDIS_QUEUE)                                                     # queue_name, msg <- 
        # print(f'Message from user: {msg}')
        
        # 2. Decode image_name
        msg_dict = json.loads(msg)
        img_name = msg_dict['image_name']
        job_id =  msg_dict['id']
        
        # 3. Predict
        predict_bboxes(img_name)
        
        pred_dict = {
                    "mAP": "[TO BE IMPLEMENTED]",
                    }
        
        # 4. Store in Redis
        db.set(job_id,json.dumps(pred_dict))

        # Don't forget to sleep for a bit at the end
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
