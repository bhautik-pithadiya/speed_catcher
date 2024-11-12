from collections import defaultdict, deque
import json
import cv2
import numpy as np
import supervision as sv  # Assuming `sv` is the library for the `get_video_frames_generator` function

# Set paths and parameters
SOURCE_VIDEO_PATH = "/media/hlink/hd/vehical_test_videos/new_test_video/test_file_1_clipped_30.mp4"
TARGET_VIDEO_PATH = "video_results/vehicles-result_1.mp4"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_NAME = "yolov8x.pt"
MODEL_RESOLUTION = 1280
SPEED_LIMIT = 40

# Initialize variables to store the polygon points
polygon_points = []

# Define a mouse callback function to capture clicks
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw a small circle to mark the point
        cv2.imshow("Frame", frame)

# Get the first frame from the video
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
frame_iterator = iter(frame_generator)
frame = next(frame_iterator).copy()  # Capture a copy of the first frame

# Show the frame and set up mouse click capturing
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.imshow("Frame", frame)
cv2.setMouseCallback("Frame", mouse_callback)

print("Click on four points in the frame to define the polygon. Press 'q' when done.")

# Wait for the user to select 4 points and press 'q'
while True:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord('q') and len(polygon_points) == 4:
        break

cv2.destroyAllWindows()

# Convert collected points to numpy array
SOURCE = np.array(polygon_points)

# Prompt the user for target width and height
TARGET_WIDTH = int(input("Enter the target width: "))
TARGET_HEIGHT = int(input("Enter the target height: "))

# Define the target rectangle
TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])
print('Displaying the image')
# Annotate and display the polygon on the first frame
annotated_frame = frame.copy()
annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=SOURCE, color=sv.Color(255, 0, 0), thickness=4)
sv.plot_image(annotated_frame)
# Display annotated frame (optional)
# cv2.imshow("Annotated Frame", annotated_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Proceed with the rest of your processing here
