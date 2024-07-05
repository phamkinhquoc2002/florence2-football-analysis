from ultralytics import YOLO
import supervision as sv
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
        detections = self.detect(frames)
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
                bbox = tracked_detection[0].to_list()
                class_id = tracked_detection[3]
                track_id = tracked_detection[4]
                
                if class_id == cls_names_inv['player']:
                    tracks["players"][id][track_id] = {"bbox":bbox}
                elif class_id == cls_names_inv["referees"]:
                    tracks["referees"][id][track_id] = {"bbox":bbox}
                    
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].to_list()
                class_id = frame_detection[3]
                
                if class_id == cls_names_inv['ball']:
                    tracks['ball'][id][track_id] = {"bbox":bbox}
                    
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)