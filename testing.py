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
    global frame, frame_copy  # Use a global frame copy to manage temporary drawing

    # If the left mouse button is clicked, register the point
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), 2)  # Draw a circle to mark the point on the permanent frame

        # Draw a line between the last two points in the permanent frame
        if len(polygon_points) >= 2:
            cv2.line(frame, polygon_points[-1], polygon_points[-2], (0, 255, 0), 2)

        # If we have four points, close the polygon by connecting the last to the first point
        if len(polygon_points) == 4:
            cv2.line(frame, polygon_points[0], polygon_points[-1], (0, 255, 0), 2)

    # Refresh `frame_copy` from `frame` to add the temporary line
    frame_copy = frame.copy()

    # Draw the dynamic line from the last clicked point to the current mouse position
    if len(polygon_points) > 0:
        cv2.line(frame_copy, polygon_points[-1], (x, y), (0, 255, 0), 5)
        cv2.line(frame_copy, polygon_points[0], (x, y), (0, 255, 0), 5)
    # Show the frame with both permanent and dynamic drawings
    cv2.imshow("Frame", frame_copy)
        

# Get the first frame from the video
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
frame_iterator = iter(frame_generator)
frame = next(frame_iterator).copy()  # Capture a copy of the first frame
frame_copy = frame.copy()
# Show the frame and set up mouse click capturing
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.imshow("Frame", frame_copy)
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

# sorting the polygon points as required
# B <- A
# v    ^
# C -> D

np.sort(SOURCE,axis=0)
temp = SOURCE[0].copy()
SOURCE[0] = SOURCE[2]
SOURCE[2] = temp


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

