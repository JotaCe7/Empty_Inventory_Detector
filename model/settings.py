import os
from enum import Enum

# Images Stored
UPLOAD_FOLDER = "uploads/"                              # Loaded by user
PREDICTIONS_FOLDER = "predictions/"                      # Predicted by model

# Target image size
TARGET_IMSIZE = 640
# Non-Max Suppression overlapping threshold
OVERLAP_THRESH = 0.7

# S3 models
MODELS_FOLDER_S3 = "anyoneai-datasets/trained_models"
BEST_MODEL = "bestv2.pt" 

# CLASSES
class CLASSES(Enum):
  PRODUCT = 3
  MISSING = 2

# COLORMAP PER CLASS
class COLORMAPS(Enum):
  PRODUCT = 'COLORMAP_TURBO'
  MISSING = 'COLORMAP_RAINBOW'

# REDIS
# Queue name
REDIS_QUEUE = "service_queue"
# Port
REDIS_PORT = 6379
# DB Id
REDIS_DB_ID = 0
# Host IP
REDIS_IP = os.getenv("REDIS_IP", "redis")
# Sleep parameters which manages the
# interval between requests to our redis queue
SERVER_SLEEP = 0.05
