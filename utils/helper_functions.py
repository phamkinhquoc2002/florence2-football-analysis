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