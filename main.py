from utils import save, read_video, Tracker

def main():
    # Read the video input
    video_frames = read_video('inputs/100.mp4')
    tracker = Tracker('models/best.pt')
    tracks = tracker.object_track(frames=video_frames,
                                  read_from_stub_path=True,
                                  stub_path='stubs/track_stubs.pkl')
    #Save the video output
    #output = save(video_frames, "outputs/output.avi")
    
if __name__ == "__main__":
    main()