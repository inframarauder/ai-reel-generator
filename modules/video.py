# pylint: disable=all 
'''
utility functions related to video processing
pylint is disabled due to weird false warnings while using cv2
'''
import time 

import cv2
import numpy as np
from tqdm import trange
from PIL import Image
from moviepy import VideoFileClip,AudioFileClip,CompositeVideoClip, concatenate_videoclips, vfx

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

def find_nearest_beat_to_timestamp(timestamp, beat_timestamps, max_shift_dist = 1):
    '''
    Method to find closest beat within range.
    '''

    # use binary search to get the closest candidate beats
    idx = np.searchsorted(beat_timestamps, timestamp)
    candidates = []
    
    # Check adjacent beats
    if idx > 0:
        candidates.append(int(beat_timestamps[idx-1]))
    if idx < len(beat_timestamps):
        candidates.append(int(beat_timestamps[idx]))
    
    # return the original timestamp if no beats found nearby
    if not candidates:
        return timestamp
    
    # choose the lower value of the two candidates - closest beat to the timestamp
    closest = min(candidates, key=lambda x: abs(x - timestamp))

    # return the closest beat_timestamp if its less than max_shift_distance
    # else return the original timestamp
    return closest if abs(closest - timestamp) <= max_shift_dist else timestamp

def sync_cuts_to_nearest_beat(video_segments, beat_timestamps):
    '''
    shift video cut-points to the nearest beat_timestamp
    (if possible)
    '''
    synced_segments = []

    for _, cut_points, video_path in video_segments:
        start, end = cut_points

        new_start = find_nearest_beat_to_timestamp(start, beat_timestamps) 
        new_end = find_nearest_beat_to_timestamp(end, beat_timestamps)

        print(f"Beat-synced segment for {video_path} - {(new_start, new_end)}")
        synced_segments.append((video_path, (new_start, new_end)))
    
    return synced_segments

def render_reel(video_segments, output_folder, retain_audio, audio_path):
    '''
        merge clips, apply transitions,sync to audio,
        do whatever the heck needs to be done and 
        render final reel clip at output_path
    '''
    # extract the required clips
    clips = []
    print("Processing top match clips ...")
    for video_path, cut_points in video_segments:

        video = VideoFileClip(video_path, audio=retain_audio) 
        start , end = cut_points

        # Cut and collect the segments with transition
        clip = video.subclipped(start,end)
        clips.append(clip)

        # close clip to free up resources
        clip.close()
    
    # stitch them with transition into a single video
    final_video = concatenate_videoclips(
        clips = clips, 
        method="compose",
        padding = -1
    )

    # add audio to final_video
    audio = AudioFileClip(audio_path).subclipped(0,final_video.duration)
    final_video = final_video.with_audio(audio)
    
    # render final video
    output_path = f"{output_folder}/reel_{time.time()}.mp4"
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=24,
        bitrate="8000k",
        threads=4
    )

     # Close final_video
    final_video.close()
