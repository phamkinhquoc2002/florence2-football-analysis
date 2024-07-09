from utils import save, read_video, Tracker, TeamAssigner
import cv2 as cv

def main():
    # Read the video input
    video_frames = read_video('inputs/inputs.mp4')
    tracker = Tracker('models/best.pt')
    tracks = tracker.object_track(frames=video_frames,
                                  read_from_stub_path=True,
                                  stub_path='stubs/track_stubs.pkl')
    
    teamAssigner = TeamAssigner()
    teamAssigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = teamAssigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = teamAssigner.team_colors[team]
    #Save player images
    #for track_id, player in tracks['players'][0].items():
        #bbox = player['bbox']
        #frame = video_frames[0]
        
        #cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        #cv.imwrite(f'outputs/cropped_img.jpg', cropped_image)
    #Save the video output
    output = tracker.draw_annotations(video_frames, tracks)
    
    save(output, "outputs/output.avi")
    
if __name__ == "__main__":
    main()