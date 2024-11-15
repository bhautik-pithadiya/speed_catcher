# logging_config.py
import logging
import os
from datetime import datetime

# Create a logger instance (this will be used across the project)
logger = logging.getLogger("root")
# Set the lowest level of logs that will be captured (DEBUG will capture all levels)
logger.setLevel(logging.DEBUG)

# Create handlers for both file and terminal
log_filename = f"logs/vehicle_tracking_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

file_handler = logging.FileHandler(log_filename)  # Log file
file_handler.setLevel(logging.INFO)  # Only capture WARNING and above for file


stream_handler = logging.StreamHandler()  # Terminal (console)
stream_handler.setLevel(logging.DEBUG)  # Capture DEBUG and above for console


# Set formatter for the log entries (for both file and console)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Assign formatter to both handlers
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)