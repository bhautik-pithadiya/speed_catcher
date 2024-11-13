# config.py

import os

# File paths
SOURCE_VIDEO_PATH = os.getenv("SOURCE_VIDEO_PATH", "/media/hlink/hd/vehical_test_videos/new_test_video/test_file_1_clipped.mp4")
TARGET_VIDEO_PATH = "video_results/vehicles-result_1.mp4"

# YOLO model parameters
MODEL_NAME = "yolov8x.pt"
MODEL_RESOLUTION = 1280
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
SPEED_LIMIT = 40


