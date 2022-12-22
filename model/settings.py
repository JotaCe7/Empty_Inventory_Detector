import os

# Images Stored
UPLOAD_FOLDER = "uploads/"                              # Loaded by user
PREDICTIONS_FOLDER = "predictions/"                      # Predicted by model

# Non-Max Suppression overlapping threshold
OVERLAP_THRESH = 0.7

# S3 models
MODELS_FOLDER_S3 = "anyoneai-datasets/trained_models"
BEST_MODEL = "best.pt" 

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
