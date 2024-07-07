import cv2

def read_video(path: str):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        isTrue, frame = cap.read()
        if not isTrue:
            break
        frames.append(frame)
    return frames

def save(output_frames, path: str):
    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(path, fourcc, 24, (output_frames[0].shape[1], output_frames[0].shape[0]))
    for frame in output_frames:
        output.write(frame)
    output.release()
    
# Calculation

def get_center_of_box(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2)/2), int((y1 + y2)/2)

def get_width_of_box(bbox):
    return int(bbox[2]-bbox[0])