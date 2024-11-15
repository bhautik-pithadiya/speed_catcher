# main.py


import os
import cv2
from datetime import datetime
import numpy as np
from tqdm import tqdm
import supervision as sv
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from logging_config import logger  # Import the globally configured logger

from config import SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH,CONFIDENCE_THRESHOLD,MODEL_RESOLUTION,IOU_THRESHOLD,SPEED_LIMIT
logger.info('Imported config successfully')
from detection import detecting_number_plate_ocr,ViewTransformer,loading_models
logger.info('Import from detection.py done successfully')
from database import insert_number_plate, create_database
logger.info('Import from database.py done successfully')
from polygon_setup import get_polygon_points
logger.info('get_polygon_points Importrd from polygon_setup.py ')
from utils import sort_points_by_x,get_splitted_polygon_points,calculate_avg_batch_speed,has_crossed_threshold




def initialize_video_processing():
    logger.info('Initialzing video processing ...')

    # Get video info
    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

    # Testing flag
    testing = True if input("testing? Yes/No")[0].lower() == 'y' else False

    # Get the first frame from the video
    frame = next(frame_generator)
    
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
    threshold_corr = get_splitted_polygon_points(SOURCE)
    logger.info(f'threshold corrs : {threshold_corr}')
    annotated_frame = frame.copy()

    # Draw the source polygon
    annotated_frame = sv.draw_polygon(
        scene=annotated_frame, 
        polygon=SOURCE, 
        color=sv.Color(255, 0, 0), 
        thickness=4
    )
    
    # Plot the threshold line (corr5 to corr6)
    annotated_frame = cv2.line(
        annotated_frame, 
        tuple(threshold_corr[0]),  # Start point (corr5)
        tuple(threshold_corr[1]),  # End point (corr6)
        color=(0, 255, 0),         # Green color
        thickness=2
    )

    # Optionally label the threshold points
    cv2.putText(annotated_frame, "corr5", tuple(threshold_corr[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(annotated_frame, "corr6", tuple(threshold_corr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    sv.plot_image(annotated_frame)
        
    # Create byte tracker
    byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_thresh=CONFIDENCE_THRESHOLD)
    
    # Creating Polygon zone 
    polygon_zone = sv.PolygonZone(
    polygon=SOURCE,
    frame_resolution_wh=video_info.resolution_wh)

    return video_info, frame_generator, byte_track,SOURCE,TARGET, polygon_zone,threshold_corr

def process_ocr_task(tracker_id, GOT_model, tokenizer,avg_speed):
    image_path = f'data/overspeeding_vehicles_ss/overspeeding_vehicle_{tracker_id}.png'
    
    # Run OCR
    plate_number = detecting_number_plate_ocr(image_path, GOT_model, tokenizer)
    print(f"OCR result for vehicle {tracker_id}: {plate_number}")
    insert_number_plate('vehicle_plates.db',tracker_id,plate_number,avg_speed,image_path)
    return tracker_id, image_path, plate_number

def track_and_calculate_speed(detections, coordinates, video_info, processed_vehicles, frame, speed_records, speed_batch_records, threshold_line, GOT_model, tokenizer, ocr_executor):
    
    labels = []
    coord5, coord6 = threshold_line  # Unpack the coordinates of the threshold line

    for tracker_id, (x_min, y_min, x_max, y_max) in zip(detections.tracker_id, detections.xyxy):
        try:
            # Calculate the vehicle's bottom center point (y-coordinate)
            vehicle_center_x = (x_min + x_max) / 2
            vehicle_center_y = (y_min + y_max) / 2
            
            # Calculate the current speed for the vehicle
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                # Calculate speed
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6  # Speed in km/h
                
                # Store the speed for every frame
                speed_records[tracker_id].append(speed)
                
                # Batch every 5 frames and calculate the average speed for the batch
                if len(speed_records[tracker_id]) % 5 == 0:  # After every 5 frames
                    avg_speed_batch = sum(list(speed_records[tracker_id])[-5:]) / 5
                    speed_batch_records[tracker_id].append(avg_speed_batch)
                
                # Check if the vehicle crosses the threshold line
                if has_crossed_threshold(vehicle_center_x, vehicle_center_y, coord5, coord6):
                    
                    # If the vehicle has already been processed, skip it
                    if tracker_id in processed_vehicles:
                        continue
                    
                    # Calculate the average speed of the batches so far
                    avg_speed_of_batches = calculate_avg_batch_speed(tracker_id, speed_batch_records)
                    
                    # Check if the average speed exceeds the speed limit
                    if avg_speed_of_batches > SPEED_LIMIT:
                        vehicle_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                        image_path = f'data/overspeeding_vehicles_ss/overspeeding_vehicle_{tracker_id}.png'
                        cv2.imwrite(image_path, vehicle_image)

                        # Run OCR on the vehicle image
                        ocr_executor.submit(process_ocr_task, tracker_id, GOT_model, tokenizer, avg_speed_of_batches)
                    
                        # Mark this vehicle as processed
                        processed_vehicles.add(tracker_id)

                labels.append(f"#{tracker_id} {int(speed)} km/h")  # Show the speed on the label

        except Exception as e:
            logger.error(f"Error processing tracker_id {tracker_id} in track_and_calculate_speed: {e}", exc_info=True)
            continue  # Skip to the next vehicle if there is an error with this one

    return labels


def process_frame(frame, model, byte_track, polygon_zone, coordinates,view_transformer):
    # Detect objects in the frame
    result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False,half=True,classes = [2])[0]
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

def save_overspeeding_vehicle(frame, tracker_id, x_min, y_min, x_max, y_max, speed, overspeeding_vehicles,GOT_model,tokenizer):
    # Save the image for OCR processing
    vehicle_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
    image_paths = f'data/overspeeding_vehicles_ss/overspeeding_vehicle_{tracker_id}.png'
    cv2.imwrite(image_paths, vehicle_image)
    
    # Run OCR (e.g., using detecting_number_plate_ocr)
    plate_number = detecting_number_plate_ocr(image_paths,GOT_model,tokenizer)
    print(f"OCR result for vehicle {tracker_id}: {plate_number}")
    
    # Store overspeeding vehicle info
    overspeeding_vehicles[tracker_id].append(((x_min, y_min, x_max, y_max), speed, image_paths))

def annotate_frame(frame, detections, labels, trace_annotator, bounding_box_annotator, label_annotator,threshold_line):
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame, detections=detections
        )
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame, detections=detections
        )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)
    
    # Draw the threshold line (corr5 to corr6)
    corr5, corr6 = threshold_line  # Unpack the threshold coordinates
    cv2.line(
        annotated_frame,
        tuple(corr5),  # Start point
        tuple(corr6),  # End point
        color=(0, 255, 0),  # Green color for the threshold line
        thickness=2,
    )

    # Optional: Label the threshold points for debugging
    cv2.putText(annotated_frame, "Threshold Line", tuple(corr5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    
    return annotated_frame

def main():
    try:   
        logger.info('Running the main.py')
        model,tokenizer,GOT_model = loading_models()
        
        create_database()
        
        video_info, frame_generator, byte_track, SOURCE,TARGET,polygon_zone,threshold_corr = initialize_video_processing()
        
        coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
        speed_records = defaultdict(lambda: deque(maxlen=5))  # {maxlen}-frame window
        speed_batch_records = defaultdict(list)
        # overspeeding_vehicles = defaultdict(list)
        processed_vehicles = set()


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
        start_timer = datetime.now()

        with ThreadPoolExecutor(max_workers=4) as ocr_executor:
            with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
                for frame in tqdm(frame_generator, total=video_info.total_frames):
                    
                    detections, coordinates = process_frame(
                        frame, 
                        model, 
                        byte_track, 
                        polygon_zone, 
                        coordinates, 
                        view_transformer)
                    
                    labels = track_and_calculate_speed(
                        detections, 
                        coordinates, 
                        video_info, 
                        processed_vehicles, 
                        frame,
                        speed_records,
                        speed_batch_records,
                        threshold_corr,
                        GOT_model,
                        tokenizer,
                        ocr_executor)
                    
                    annotated_frame = annotate_frame(
                        frame, 
                        detections, 
                        labels, 
                        trace_annotator, 
                        bounding_box_annotator, 
                        label_annotator,
                        threshold_corr)
                    
                    sink.write_frame(annotated_frame)
        end_timer = datetime.now()
        logger.info(f'Time to complete the video processing - {end_timer-start_timer}')
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise  # Re-raise to stop execution if necessary

if __name__ == "__main__":
    
    main()
    
    