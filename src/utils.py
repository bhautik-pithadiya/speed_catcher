# utils.py

import numpy as np
import time
from logging_config import logger


def sort_points_by_x(points):
    try:
        if not points:
            raise ValueError('Polygon points may be empty')
        logger.info('Staring the sort_points_by_x')
        return points[points[:, 0].argsort()]
    except ValueError as e:
        logger.error(f"ValueError occurred: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f'Error occurred in sort_points_by_x function:{e}',exc_info=True)
        raise

def get_splitted_polygon_points(points):
    try:
        if points.size==0:
            raise ValueError('Polygon points may be empty')
        logger.info("Getting points from get_splitted_polygon_points function")
        # define the fraction 40% -> 0.4
        t = 0.3
        corr2,corr3 = points[1].copy(),points[2].copy()
        corr1,corr4 = points[0].copy(),points[3].copy()
        
        corr5 = np.round(corr3 + t*(corr2-corr3)).astype(int)
        corr6 = np.round(corr4 + t*(corr1-corr4)).astype(int)
        
        all_points = np.append(points,np.array([corr5,corr6]),axis=0)
        logger.info(f'\n{all_points}')
        return [corr5,corr6]
    except Exception as e:
        logger.error(f'Error Occurred in get_splitted_polygon_points function: {e}',exc_info=True)
        raise

def calculate_avg_batch_speed(tracker_id, speed_batch_records):
    """
    Calculate the average speed of the batches for a given vehicle.

    Parameters:
    - tracker_id: The unique ID of the vehicle.
    - speed_batch_records: Dictionary storing the average speed of every batch for each vehicle.

    Returns:
    - avg_speed: The average speed of the batches for the vehicle.
    """
    try:
        if tracker_id in speed_batch_records and len(speed_batch_records[tracker_id]) > 0:
            avg_speed = sum(speed_batch_records[tracker_id]) / len(speed_batch_records[tracker_id])
            logger.info(f"Average speed for vehicle {tracker_id}: {avg_speed} km/h")
            return avg_speed
        else:
            logger.warning(f"No speed records found for vehicle {tracker_id}. Returning 0.")
            return 0  # Return 0 if there are no batches or records for the tracker
    except Exception as e:
        logger.error(f"Error calculating average speed for vehicle {tracker_id}: {e}", exc_info=True)
        return 0  # In case of an error, return 0

def has_crossed_threshold(vehicle_center_x, vehicle_center_y, coord5, coord6):
    """
    Checks if a vehicle's center point has crossed the threshold line defined by coord5 and coord6.
    """
    x1, y1 = coord5
    x2, y2 = coord6

    # Compute the equation of the line: ax + by + c = 0
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    # Plug the vehicle's center point into the line equation
    result = a * vehicle_center_x + b * vehicle_center_y + c

    return result > 0  # Adjust condition based on coordinate system orientation
