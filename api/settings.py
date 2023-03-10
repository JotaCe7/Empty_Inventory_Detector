import os

# Run API in Debug mode
API_DEBUG_FLAG= True

# Images
#DEMO_IMGS = "demo_imgs/"                        # Store images for demo
UPLOAD_FOLDER = "static/uploads/"               # Store h5 hashed images (uploaded by user)
PREDICTIONS_FOLDER = "static/predictions/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# We will store user feedback on this file
FEEDBACK_FILEPATH = "feedback/feedback"
os.makedirs(os.path.basename(FEEDBACK_FILEPATH), exist_ok=True)

# REDIS settings
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
API_SLEEP = 0.05

# ------------------------
# - ADDED BY @sannicosan -
# Extensions allowed for image files
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
RPSE_MIMETYPE = 'application/json'