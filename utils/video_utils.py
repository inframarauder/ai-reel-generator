# pylint: disable=all 
'''
utility functions related to video processing
pylint is disabled due to weird false warnings while using cv2
'''
import time 

import cv2
from tqdm import trange
from PIL import Image
from moviepy import VideoFileClip,CompositeVideoClip, concatenate_videoclips, vfx

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
        
        # add to frames dict
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

def render_reel(video_segments, output_folder, retain_audio):
    '''
        merge clips, apply transitions,sync to audio,
        do whatever the heck needs to be done and 
        render final reel clip at output_path
    '''
    # extract the required clips
    clips = []
    print("Processing top match clips ...")
    for match_score,clip_window,video_path in video_segments:

        video = VideoFileClip(video_path, audio=retain_audio) 
        start , end = clip_window

        print(f"{video_path} - {clip_window} [{round(match_score*100)}% match]")

        # Cut and collect the segments with transition
        clip = video.subclipped(start,end)
        clips.append(clip)

        # close clip to free up resources
        clip.close()
    
    # stitch them with transition into a single video
    final_video = concatenate_videoclips(
        clips = clips, 
        method="compose",
        padding = 0
    )

    # render final video
    output_path = f"{output_folder}/reel_{time.time()}.mp4"
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",  # Required parameter even with no audio
        fps=30,
        threads=4
    )

     # Close final_video
    final_video.close()

    print(f"Reel rendered at path: {output_path}")