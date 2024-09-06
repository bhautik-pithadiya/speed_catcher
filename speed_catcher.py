import cv2

import numpy as np
import supervision as sv

from tqdm import tqdm
from ultralytics import YOLO
# from supervision.assets import VideoAssets, download_assets
from collections import defaultdict, deque


SOURCE_VIDEO_PATH = "/media/hlink/hd/vehical_test_videos/new_test_video/test_file_1_clipped_30.mp4"
TARGET_VIDEO_PATH = "vehicles-result.mp4"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_NAME = "yolov8x.pt"
MODEL_RESOLUTION = 1280


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


annotated_frame = frame.copy()
annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=SOURCE, color=sv.Color(255,0,0), thickness=4)
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
        if frame_counter % 3 == 0:  
            for tracker_id in detections.tracker_id:
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

                                        # Store the calculated speed
                    last_speeds[tracker_id] = speed
                    labels.append(f"#{tracker_id} {int(speed)} km/h")
        else:
            # Show the last calculated speed if available
            for tracker_id in detections.tracker_id:
                if tracker_id in last_speeds:
                    labels.append(f"#{tracker_id} {int(last_speeds[tracker_id])} km/h")
                else:
                    labels.append(f"#{tracker_id}")



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


