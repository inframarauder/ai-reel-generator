# pylint: disable=all 
'''
utility functions related to video processing
pylint is disabled due to weird false warnings while using cv2
'''
import cv2
import ffmpeg
from tqdm import trange
from PIL import Image

def extract_video_info_and_frames(video_path, sampling_rate):
    '''
    This method displays video information and 
    returns a list of all frames in the video
    '''
    print(f"Processing video {video_path}")

    # capture video using OpenCV
    cap = cv2.VideoCapture(video_path, apiPreference=0, params=[])

    # Get video properties
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = int(total_frames / fps) # in seconds    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

    print(f"Video Info:\nNumber of frames: {total_frames}\nFPS: {fps}\nResolution: {width} x {height}\nVideo Length: {total_duration} seconds")

    frames_to_sample = int(total_duration * sampling_rate)
    frames = dict()

    print(f"Sampling {frames_to_sample} frames out of {total_frames} ({sampling_rate} frames per second) ...")
    for iter in trange(frames_to_sample):
        # read frame
        success, frame = cap.read()

        # break loop if unable to read frame
        if not success:
            print("Failed to read frame!")
            break
        
        # get current timestamp in seconds
        timestamp_sec = int(iter/sampling_rate)

        # convert current frame to a PIL image for ease of vector-embedding
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # add to frames dict and increase frame_count
        frames[timestamp_sec] = {
            "pil_image" : pil_image,
            "opencv_frame" : frame
        }
    
    return {
       "fps" : fps,
       "total_frames": total_frames,
       "total_duration": total_duration,
       "width" : width,
       "height": height,
       "frames":frames
    }

def render_reel_video(video_path,output_path,start,end,retain_audio):
    """
    Cut video using ffmpeg-python.
    """
    try:
       return (
            ffmpeg
            .input(video_path, ss=start, to=end, )
            .output(output_path, vcodec="copy", an=None)
            .overwrite_output()
            .run(quiet = True)
        ) if retain_audio else (
            ffmpeg
            .input(video_path, ss=start, to=end)
            .output(output_path, vcodec="copy")
            .overwrite_output()
            .run(quiet = True)
        )
    except ffmpeg.Error as e:
        print(f"Error cutting video: {e.stderr.decode('utf8')}")
        raise