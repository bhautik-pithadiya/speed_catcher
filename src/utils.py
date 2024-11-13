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
        t = 0.4
        corr2,corr3 = points[1].copy(),points[2].copy()
        corr1,corr4 = points[0].copy(),points[3].copy()
        
        corr5 = np.round(corr3 + t*(corr2-corr3)).astype(int)
        corr6 = np.round(corr4 + t*(corr1-corr4)).astype(int)
        
        all_points = np.append(points,np.array([corr5,corr6]),axis=0)
        # points.append(corr6)
        logger.info(f'\n{all_points}')
        time.sleep(15)
        return all_points
    except Exception as e:
        logger.error(f'Error Occurred in get_splitted_polygon_points function: {e}',exc_info=True)
        raise