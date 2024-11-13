# detection.py
from logging_config import logger  # Import the globally configured logger
import cv2
import numpy as np
import supervision as sv
from collections import defaultdict,deque
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from ultralytics import YOLO
from config import SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH, MODEL_NAME, MODEL_RESOLUTION, CONFIDENCE_THRESHOLD, IOU_THRESHOLD

# Video and model setup
# video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

def loading_models():
    try:
        logger.info('Model Loading Started')
        model = YOLO(MODEL_NAME)
        logger.info('Yolo model loaded')
        tokenizer = AutoTokenizer.from_pretrained("ucaslcl/GOT-OCR2_0", trust_remote_code=True)
        GOT_model = AutoModel.from_pretrained("ucaslcl/GOT-OCR2_0", trust_remote_code=True, device_map='cuda')
        logger.info('GOT model and its tokenizer loaded')
        return model,tokenizer,GOT_model
    except Exception as e:
        logger.error(f'Error while loading the model{e}',exc_info=True)
        raise



# Define a class for perspective transformation
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target) 

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def detecting_number_plate_ocr(image_path, model, tokenizer):
    try:
        if not image_path or not model or not tokenizer:
            raise ValueError("Invalid input: image_path, model, or tokenizer is None or empty.")
        logger.info("GOT model doing it's work")
        res = model.chat(tokenizer, image_path, ocr_type='ocr')
        if not res:
            raise ValueError('Invalid output: model returned None or empty')
        return res
    except ValueError as e:
        logger.error(f"ValueError occurred: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f'Error occurred while using detecting_number_plate_ocr method{e}',exc_info=True)
        raise
        
