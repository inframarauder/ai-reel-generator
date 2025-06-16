# pylint: disable=all

import cv2

def extract_video_frames(video_path):
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
    frames = []
    while True:
        # read frame
        success, frame = cap.read()

        # break loop if unable to read frame
        if not success:
            print("Failed to read frame!")
            break
        
        # add frame to list of frames and increase frame_count
        frames.append(frame)
        frame_count += 1
        print(f"Processed {frame_count}/{total_frames} frames..")
    
    return frames



def get_clip_window(prompt_emb, video_path, k):
    video_frames = extract_video_frames(video_path)
    timestamp_embedding_map = create_timestamp_embedding_map(video_frames)

    timestamps = timestamp_embedding_map.keys()
    image_embs = timestamp_embedding_map.values()

    clip_window = (0,0)
    n = len(timestamp_embedding_map.items())
    i = 0
    avg_similarity = 0

    while i <= (n-k):
        timestamp_window = timestamps[i:i+k]
        image_emb_window = image_embs[i:i+k]
        image_similarity_window = [cosine_similarity(prompt_emb, image_emb) for image_emb in image_emb_window]

        curr_sum = sum(image_similarity_window)
        curr_avg = curr_sum / k

        if curr_avg > avg_similarity:
            avg_similarity = curr_avg
            i = k
            clip_window = (timestamp_window[0], timestamp_window[len(timestamp_window) - 1])
            
        else:
            i = i + 1

        return clip_window