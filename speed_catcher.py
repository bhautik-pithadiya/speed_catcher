from collections import defaultdict, deque
import json
import cv2

import os
import numpy as np
import supervision as sv

from tqdm import tqdm
from ultralytics import YOLO

from transformers import AutoModel, AutoTokenizer
import sqlite3
from datetime import datetime

# from supervision.assets import VideoAssets, download_assets
import sqlite3

def create_database():
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('vehicle_plates.db')
    c = conn.cursor()

    # Create table to store vehicle number plate data
    c.execute('''CREATE TABLE IF NOT EXISTS vehicle_plates (
                    id INTEGER PRIMARY KEY,
                    vehicle_id INTEGER,
                    plate_number TEXT,
                    timestamp TEXT,
                    speed REAL,
                    image_path TEXT
                  )''')
    conn.commit()
    conn.close()
    
create_database()

def insert_number_plate(vehicle_id, plate_number, speed, image_path):
    conn = sqlite3.connect('vehicle_plates.db')
    c = conn.cursor()

    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Check if the vehicle ID already exists in the database
    c.execute('''SELECT * FROM vehicle_plates WHERE vehicle_id = ?''', (vehicle_id,))
    existing_record = c.fetchone()
    
    if existing_record:
        # If record exists, update the speed and timestamp
        c.execute('''UPDATE vehicle_plates 
                     SET speed = ?, timestamp = ?, plate_number = ? 
                     WHERE vehicle_id = ?''', 
                     (speed, timestamp, plate_number,vehicle_id))
    else:
        # If record doesn't exist, insert a new one
        c.execute('''INSERT INTO vehicle_plates (vehicle_id, plate_number, timestamp, speed, image_path)
                     VALUES (?, ?, ?, ?, ?)''', 
                     (vehicle_id, plate_number, timestamp, speed, str(image_path)))

    conn.commit()
    conn.close()

def fetch_all_data():
    conn = sqlite3.connect('vehicle_plates.db')
    c = conn.cursor()

    c.execute("SELECT * FROM vehicle_plates")
    rows = c.fetchall()

    for row in rows:
        print(row)

    conn.close()


os.environ["SOURCE_VIDEO_PATH"]  = "/media/hlink/hd/vehical_test_videos/new_test_video/test_file_1_clipped_30.mp4"
SOURCE_VIDEO_PATH = os.getenv("SOURCE_VIDEO_PATH")
if not os.path.exists(SOURCE_VIDEO_PATH):
    raise Exception(f"File not found at {SOURCE_VIDEO_PATH}")

TARGET_VIDEO_PATH = "video_results/vehicles-result_1.mp4"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_NAME = "yolov8x.pt"
MODEL_RESOLUTION = 1280
SPEED_LIMIT = 40
testing = True if input("testing? Yes/No")[0].lower() == 'y' else False

if testing == False:
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

    ## Convert collected points to numpy array
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
else:
    
    SOURCE = np.array([
        [1100,410],
        [715,410],
        [200,1010],
        [1430,1010]
    ])

    TARGET_WIDTH = 15
    TARGET_HEIGHT = 90

    TARGET = np.array([
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ])
    frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
    frame_iterator = iter(frame_generator)
    frame = next(frame_iterator)
    
    # annotated_frame = frame.copy()
    # annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=SOURCE, color=sv.Color(255,0,0), thickness=4)
    # sv.plot_image(annotated_frame)
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


view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

import cv2

model = YOLO(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained("ucaslcl/GOT-OCR2_0", trust_remote_code=True)
GOT_model = AutoModel.from_pretrained("ucaslcl/GOT-OCR2_0",trust_remote_code=True, device_map='cuda')

def detecting_number_plate_ocr(image_path,model=GOT_model,tokenizer=tokenizer):
    res = model.chat(tokenizer,image_path,ocr_type = 'ocr')
    return res

video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

# tracer initiation
byte_track = sv.ByteTrack(
    frame_rate=video_info.fps,  track_thresh=CONFIDENCE_THRESHOLD
)

# annotators 
thickness = 1
# thickness = sv.draw.utils.calculate_dynamic_line_thickness(
#     resolution_wh=video_info.resolution_wh
# )

text_scale = 1
# text_scale = sv.draw.utils.calculate_dynamic_text_scale(
#     resolution_wh=video_info.resolution_wh
# )
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

polygon_zone = sv.PolygonZone(
    polygon=SOURCE,
    frame_resolution_wh=video_info.resolution_wh
)

coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
speed_records = defaultdict(lambda: deque(maxlen=3))  # 3-frame window
overspeeding_vehicles = defaultdict(list)  # Store vehicle ID, bounding box, and speed
frame_counter = 0  # Counter to keep track of frames
last_speeds = {}   # Dictionary to store the last calculated speed for each tracker ID

with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:

    # loop over source video frames
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        frame_counter += 1

        result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # filter out detections by class and confidence
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        detections = detections[detections.class_id != 0]

        # filter out detections outside the zone
        detections = detections[polygon_zone.trigger(detections)]

        # refine detections using non-max suppression
        detections = detections.with_nms(IOU_THRESHOLD)

        # pass detection through the tracker
        detections = byte_track.update_with_detections(detections=detections)

        # points = detections.get_anchors_coordinates(
        points = detections.get_anchor_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )

        # calculate the detections position inside the target RoI
        points = view_transformer.transform_points(points=points).astype(int)

        # store detections position
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        # format labels
        labels = []
        # print(poin)
        # if frame_counter % 3 == 0:  
        for tracker_id,(x_min, y_min, x_max, y_max)in zip(detections.tracker_id, detections.xyxy):
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                # calculate speed                          
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
                        print(f"Vehicle {tracker_id} is overspeeding!")
                        # x_min, y_min, x_max, y_max = coordinate_end
                        # Extract the vehicle's region from the current frame
                        vehicle_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                        # Save the image for OCR processing
                        image_paths = f'data/overspeeding_vehicles_ss/overspeeding_vehicle_{tracker_id}.png'
                        cv2.imwrite(f"data/overspeeding_vehicles_ss/overspeeding_vehicle_{tracker_id}.png", vehicle_image)

                        overspeeding_vehicles[tracker_id].append(((x_min, y_min, x_max, y_max), avg_speed,image_paths))

        # # print(overspeeding_vehicles.items())
        for tracker_id, x in overspeeding_vehicles.items():
            # Extract bounding box coordinates
            # print(x,x[0][-1])
            speed = x[0][1]
            image_path = x[0][-1]
            # x_min, y_min, x_max, y_max = bbox  # Get the latest bounding box

            # Extract the vehicle's region from the current frame
            # vehicle_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

            # Save the image for OCR processing
            # cv2.imwrite(f"overspeeding_vehicles_ss/overspeeding_vehicle_{tracker_id}.png", vehicle_image)


            # Optionally, run your OCR system here on the saved image
            plate_number = detecting_number_plate_ocr(f"data/overspeeding_vehicles_ss/overspeeding_vehicle_{tracker_id}.png")
            print(f"OCR result for vehicle {tracker_id}: {plate_number}")
            print(f"Updating vehicle {tracker_id} with speed {speed}")
            insert_number_plate(tracker_id, plate_number, speed, image_path)
            
                


        # annotate frame
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        # add frame to target video
        sink.write_frame(annotated_frame)

    
        # Optionally, run your OCR system here on the saved image
        # # ocr_result = run_ocr(f"overspeeding_vehicle_{tracker_id}.png")
        # print(f"OCR result for vehicle {tracker_id}: {ocr_result}")

# with open("sample.json", "w") as outfile: 
#     json.dump(speed_ids, outfile)
