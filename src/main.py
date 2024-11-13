# main.py


import os
import cv2
import numpy as np
from tqdm import tqdm
import supervision as sv
from collections import defaultdict, deque
from logging_config import logger  # Import the globally configured logger

from config import SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH,CONFIDENCE_THRESHOLD,MODEL_RESOLUTION,IOU_THRESHOLD,SPEED_LIMIT
logger.info('Imported config successfully')
from detection import detecting_number_plate_ocr,ViewTransformer,loading_models
logger.info('Import from detection.py done successfully')
from database import insert_number_plate, create_database
logger.info('Import from database.py done successfully')
from polygon_setup import get_polygon_points
logger.info('get_polygon_points Importrd from polygon_setup.py ')
from utils import sort_points_by_x,get_splitted_polygon_points




def initialize_video_processing():
    logger.info('Initialzing video processing ...')
    # Get video info
    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
    
    # Create byte tracker
    byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_thresh=CONFIDENCE_THRESHOLD)
    
    # Create polygon zone using the polygon points
    polygon_points = get_polygon_points(next(frame_generator))  # Get polygon points from the first frame
    polygon_zone = sv.PolygonZone(polygon=polygon_points, frame_resolution_wh=video_info.resolution_wh)

    return video_info, frame_generator, byte_track,polygon_zone

def track_and_calculate_speed(detections, coordinates, video_info, overspeeding_vehicles, frame,speed_records):
    # logger.info('Tracking vehicle and calculating speed of the individual vehicles....')
    labels = []
    for tracker_id, (x_min, y_min, x_max, y_max) in zip(detections.tracker_id, detections.xyxy):
        if len(coordinates[tracker_id]) < video_info.fps / 2:
            labels.append(f"#{tracker_id}")
        else:
            # Calculate speed
            coordinate_start = coordinates[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            distance = abs(coordinate_start - coordinate_end)
            time = len(coordinates[tracker_id]) / video_info.fps
            speed = distance / time * 3.6
            
            labels.append(f"#{tracker_id} {int(speed)} km/h") 
            speed_records[tracker_id].append(speed)

            # Check if vehicle is overspeeding over the last 3 speeds
            if len(speed_records[tracker_id]) == 3:
                avg_speed = sum(speed_records[tracker_id]) / 3
                if avg_speed > SPEED_LIMIT:
                    save_overspeeding_vehicle(frame, tracker_id, x_min, y_min, x_max, y_max, avg_speed, overspeeding_vehicles)

    return labels

def process_frame(frame, model, byte_track, polygon_zone, coordinates,view_transformer):
    # Detect objects in the frame
    result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    
    # Filter out detections by class and confidence
    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
    detections = detections[detections.class_id != 0]
    
    # Filter out detections outside the polygon zone
    detections = detections[polygon_zone.trigger(detections)]

    # Refine detections using non-max suppression
    detections = detections.with_nms(IOU_THRESHOLD)

    # Track the detections
    detections = byte_track.update_with_detections(detections=detections)

    points = detections.get_anchor_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    
    # Store the positions of the detections
    points = view_transformer.transform_points(points=points).astype(int)
    for tracker_id, [_, y] in zip(detections.tracker_id, points):
        coordinates[tracker_id].append(y)

    return detections, coordinates

def save_overspeeding_vehicle(frame, tracker_id, x_min, y_min, x_max, y_max, speed, overspeeding_vehicles):
    # Save the image for OCR processing
    vehicle_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
    image_paths = f'overspeeding_vehicles_ss/overspeeding_vehicle_{tracker_id}.png'
    cv2.imwrite(image_paths, vehicle_image)
    
    # Run OCR (e.g., using detecting_number_plate_ocr)
    plate_number = detecting_number_plate_ocr(image_paths)
    print(f"OCR result for vehicle {tracker_id}: {plate_number}")
    
    # Store overspeeding vehicle info
    overspeeding_vehicles[tracker_id].append(((x_min, y_min, x_max, y_max), speed, image_paths))

def annotate_frame(frame, detections, labels, trace_annotator, bounding_box_annotator, label_annotator):
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    return annotated_frame

def main():
    try:   
        logger.info('Running the main.py')
        model,tokenizer,GOT_model = loading_models()
        create_database()

        # Testing flag
        testing = True if input("testing? Yes/No")[0].lower() == 'y' else False
        
        # Get the first frame from the video
        frame_generator = cv2.VideoCapture(SOURCE_VIDEO_PATH)
        ret, frame = frame_generator.read()
        
        if testing:
            SOURCE = np.array([
                [1100,410],
                [715,410],
                [200,1010],
                [1430,1010]
            ])

            TARGET_WIDTH = 15
            TARGET_HEIGHT = 90
        
        else:
            

            if not ret:
                raise Exception(f"Failed to read video file: {SOURCE_VIDEO_PATH}")

            # Get the polygon points from the user click on the first frame
            SOURCE = get_polygon_points(frame)
            
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
        # print(SOURCE)
        logger.info('Displaying the image')
        # Annotate and display the polygon on the first frame
        points = get_splitted_polygon_points(SOURCE)
        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=points, color=sv.Color(255, 0, 0), thickness=4)
        sv.plot_image(annotated_frame)
        
        
        video_info, frame_generator, byte_track, polygon_zone = initialize_video_processing()
        
        coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
        speed_records = defaultdict(lambda: deque(maxlen=10))  # 10-frame window
        overspeeding_vehicles = defaultdict(list)

        view_transformer = ViewTransformer(source=SOURCE,target=TARGET)
        # annotators 
        thickness = 1
        text_scale = 1
        bounding_box_annotator = sv.BoundingBoxAnnotator(
            thickness=thickness
        )
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER
        )
        trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=video_info.fps * 2,
            position=sv.Position.BOTTOM_CENTER
        )
        
        with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
            for frame in tqdm(frame_generator, total=video_info.total_frames):
                
                detections, coordinates = process_frame(frame, model, byte_track, polygon_zone, coordinates, view_transformer)
                
                labels = track_and_calculate_speed(detections, coordinates, video_info, overspeeding_vehicles, frame,speed_records)
                
                annotated_frame = annotate_frame(frame, detections, labels, trace_annotator, bounding_box_annotator, label_annotator)
                
                sink.write_frame(annotated_frame)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise  # Re-raise to stop execution if necessary

if __name__ == "__main__":
    main()