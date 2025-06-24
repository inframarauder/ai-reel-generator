# pylint: disable=all 
'''
utility functions related to video processing
pylint is disabled due to weird false warnings while using cv2
'''
import cv2
import numpy as np
from tqdm import trange
from PIL import Image
from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.fx import FadeIn, FadeOut, CrossFadeIn

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

def get_concatenated_video(video_segments=[], video_clips =[],retain_audio = False, padding = 0):
    """
    Method to return a concatenated video
    from video_segments or video_clips
    """
    clips = []
    if len(video_clips):
        clips = video_clips
    else:
        for _, video_path, cut_points in video_segments:
            
            # load video and cut points
            video= VideoFileClip(video_path, audio=retain_audio)
            start,end = cut_points

            # extract the clip from main video
            clip = video.subclipped(start, end)

            # collect clip
            clips.append(clip)

    concatenated_video = concatenate_videoclips(
        clips=clips,
        method="compose",
        padding=padding
    )

    return concatenated_video
        

def sync_cuts_to_nearest_beat(video, cut_tempo , beat_timestamps):
    '''
    shift video cut-points to the nearest beat_timestamp
    (if possible)
    '''
    synced_clips = []
    seg_start = 0
    seg_end = cut_tempo

    for i, clip in enumerate(video.clips):

        if seg_end > len(beat_timestamps) - 1 : break

        # set start and end points of cut
        start = beat_timestamps[seg_start]
        end = beat_timestamps[seg_end]

        # set new start and end in the clip
        clip = clip.with_start(start)
        clip = clip.with_end(end)

        # apply transitions
        if i == 0:
            clip = FadeIn(1).apply(clip=clip)
        elif i == len(video.clips) - 1:
            clip = FadeOut(1).apply(clip=clip)
        else:
            clip = CrossFadeIn(1).apply(clip=clip)

        # move segment pointers
        seg_start += cut_tempo
        seg_end += cut_tempo

        # collect the newly synced clips
        synced_clips.append(clip)
    
    synced_video = get_concatenated_video(
        video_clips=synced_clips,
        padding= -1 # apparently this adds a slight fade effect
    )
    
    return synced_video
    

def render_reel(final_video, final_audio, output_folder ):
    '''
        render final reel clip at output_path
    '''
    # add audio to final_video
    final_video = final_video.with_audio(final_audio)

    # Ensure 3:4 aspect ratio (e.g., 720x960), center crop if needed
    target_w = 720
    target_h = 960

    # Resize the video to 3:4 aspect ratio (720x960)
    final_video = final_video.resized(new_size=(target_w, target_h))

    # render final video
    audio_file_name = final_audio.filename.split("/")[-1:][0].strip(".mp3")
    output_path = f"{output_folder}/reel_{audio_file_name}.mp4"

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
