# pylint: disable=all 
'''
    utility functions related to video processing
    pylint is disabled due to weird false warnings while using cv2
'''

import cv2
from PIL import Image

def extract_video_info_and_frames(video_path):
    '''
        This method displays video information and 
        returns a list of all frames in the video
    '''

    # capture video using OpenCV
    cap = cv2.VideoCapture(video_path) 

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

    print(f"Video Info:\nNumber of frames: {total_frames}\nFPS: {fps}\nResolution: {width} x {height}")

    frame_count = 0
    frames = dict()
    while True:
        # read frame
        success, frame = cap.read()

        # break loop if unable to read frame
        if not success:
            print("Failed to read frame!")
            break
        
        # get current frame's timestamp in seconds
        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # convert current frame to a PIL image for ease of vector-embedding
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # add to frames dict and increase frame_count
        frames[timestamp_sec] = {
            "pil_image" : pil_image,
            "opencv_frame" : frame
        }
        frame_count += 1
        print(f"Processed {frame_count}/{total_frames} frames..")
    
    return {
       "fps" : fps,
       "total_frames": total_frames,
       "width" : width,
       "height": height,
       "frames":frames
    }

def render_reel_video(video,output_path,retain_audio):
    # TODO: implement frame stitching to render final video
    return ""