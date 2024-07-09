from ultralytics import YOLO
from .helper_functions import get_width_of_box, get_center_of_box
import supervision as sv
import numpy as np
import cv2 as cv
import pickle
import os

class Tracker:
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect(self, frames, batch_size):
        results=[]
        for i in range(0, len(frames), batch_size):
            detected = self.model.predict(frames[i:i+batch_size], conf=0.1)
            results +=detected
        return results
    
    def object_track(self, frames, read_from_stub_path= False, stub_path=None):
        if read_from_stub_path and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks               
        detections = self.detect(frames, batch_size=20)
        tracks = {
            "goalkeeper":[],
            "players": [],
            "referees":[],
            "ball":[]
        }
        for id, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}
            #Change to Supervision format for byte tracking
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["goalkeeper"].append({})
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for tracked_detection in detection_with_tracks:
                bbox = tracked_detection[0].tolist()
                class_id = tracked_detection[3]
                track_id = tracked_detection[4]
                
                if class_id == cls_names_inv['player']:
                    tracks["players"][id][track_id] = {"bbox":bbox}
                elif class_id == cls_names_inv["referee"]:
                    tracks["referees"][id][track_id] = {"bbox":bbox}
                elif class_id == cls_names_inv["goalkeeper"]:
                    tracks["goalkeeper"][id][track_id] = {"bbox":bbox}
                    
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                
                if class_id == cls_names_inv['ball']:
                    tracks['ball'][id][track_id] = {"bbox":bbox}
                    
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
                
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_box(bbox)
        width = get_width_of_box(bbox)
        
        cv.ellipse(
            frame, 
            center=(x_center, y2),
            axes=(int(width), int(0.35*int(width))),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv.LINE_4
        )
        
        rec_width = 40
        rec_height = 20
        rec_x1 = x_center - rec_width//2
        rec_x2 = x_center + rec_width//2
        rec_y1 = (y2 - rec_height//2) + 15
        rec_y2 = (y2 + rec_height//2) + 15
        if track_id is not None:
            cv.rectangle(
                frame,
                (int(rec_x1), int(rec_y1)),
                (int(rec_x2), int(rec_y2)),
                color,
                cv.FILLED
            )
            x1_text = rec_x1 + 12
            if track_id > 99:
                x1_text -=10
            cv.putText(
                frame, 
                f"{track_id}",
                (int(x1_text), int(rec_y1+15)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                thickness=2
            )
        return frame
    
    def draw_ball(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_box(bbox)
        triangle_points = np.array(
            [
                [x, y],
                [x-10, y-20],
                [x+10, y-20],
            ]
        )
        cv.drawContours(frame, [triangle_points], 0, color, cv.FILLED)
        cv.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
        return frame    
            
    def draw_annotations(self, video_frames, tracks):
        output_frames = []
        for num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            players_dict = tracks['players'][num]
            referee_dict = tracks['referees'][num]
            ball_dict = tracks['ball'][num]
            
            #Draw players:
            for track_id, player in players_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame=self.draw_ellipse(frame, player["bbox"], color, track_id)
            for _, ref in referee_dict.items():
                frame=self.draw_ellipse(frame, ref["bbox"], (255, 255, 0))
            for _, ball in ball_dict.items():
                frame=self.draw_ball(frame, ball["bbox"], (0, 255, 0))
            output_frames.append(frame)            
        return output_frames