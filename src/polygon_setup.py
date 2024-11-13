# polygon_setup.py

import cv2
import numpy as np
from logging_config import logger  # Import the globally configured logger


polygon_points = []

# Define a mouse callback function to capture clicks
def mouse_callback(event, x, y, flags, param):
    global polygon_points
    
    frame, frame_copy = param

    # If the left mouse button is clicked, register the point
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        logger.info(f"Point Added : {polygon_points[-1]}")
        cv2.circle(frame, (x, y), 5, (0, 255, 0), 2)  # Draw a circle to mark the point on the permanent frame

        # Draw a line between the last two points in the permanent frame
        if len(polygon_points) >= 2:
            cv2.line(frame, polygon_points[-1], polygon_points[-2], (0, 255, 0), 2)

        # If we have four points, close the polygon by connecting the last to the first point
        if len(polygon_points) == 4:
            cv2.line(frame, polygon_points[0], polygon_points[-1], (0, 255, 0), 2)
            logger.info("Polygon completed. 4 points selected.")  # Log when the polygon is completed

    
    # Refresh `frame_copy` from `frame` to add the temporary line
    frame_copy = frame.copy()

    # Draw the dynamic line from the last clicked point to the current mouse position
    if len(polygon_points) > 0:
        cv2.line(frame_copy, polygon_points[-1], (x, y), (0, 255, 0), 5)
        if len(polygon_points) > 1:
            cv2.line(frame_copy, polygon_points[0], (x, y), (0, 255, 0), 5)
    
    # Show the frame with both permanent and dynamic drawings
    cv2.imshow("Frame", frame_copy)

def get_polygon_points(frame):
    global polygon_points
    polygon_points = []  # Reset the points for each call

    frame_copy = frame.copy()
    # Show the frame and set up mouse click capturing
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", mouse_callback,param=(frame,frame_copy))

    logger.info("Click on four points in the frame to define the polygon. Press 'q' when done.")

    # Wait for the user to select 4 points and press 'q'
    while True:
        key = cv2.waitKey(1)
        if key == ord('q') and len(polygon_points) == 4:
            logger.info("User pressed 'q'. Polygon selection completed.")  # Log when user finishes
            break

    cv2.destroyAllWindows()
    
    # Convert collected points to numpy array
    if len(polygon_points) != 4:
        logger.error(f"Polygon selection incomplete: {len(polygon_points)} points selected.")
        raise ValueError("Polygon selection incomplete.")
    
    SOURCE = np.array(polygon_points)

    # Sort points by x-coordinate to separate left and right
    SOURCE = SOURCE[SOURCE[:, 0].argsort()]

    # Now SOURCE[:2] are left points and SOURCE[2:] are right points
    left_points = SOURCE[:2]
    right_points = SOURCE[2:]

    # Sort left points by y-coordinate to get Bottom-Left and Top-Left
    bottom_left, top_left = left_points[np.argsort(left_points[:, 1])]

    # Sort right points by y-coordinate to get Bottom-Right and Top-Right
    bottom_right, top_right = right_points[np.argsort(right_points[:, 1])]

    # Arrange points in the desired order: Top-Right, Top-Left, Bottom-Left, Bottom-Right
    SOURCE = np.array([top_right, top_left, bottom_left, bottom_right])

    return SOURCE
