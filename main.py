'''
    This is the main.py file.
    You know what its for :)
'''

from utils.video_utils import extract_video_info_and_frames, render_reel_video
from utils.vector_utils import compute_prompt_embedding,compute_image_embedding, get_clip_window
from config import AppConfig

def main(app_config):
    '''
        main method - you know what it does :)
    '''

    # compute embeddings for the prompt
    prompt_emb = compute_prompt_embedding(
        prompt = app_config.prompt
    )

    # extract video info and frames
    video = extract_video_info_and_frames(
        video_path = app_config.video_path
    )

    # create a map of timestamps -> image embeddings for each video frame
    timestamp_embedding_map = {t: compute_image_embedding(image=img["pil_image"]) for t, img in video["frames"].items()}
   
    # get start and end times for the clip to cut out of the video
    start, end = get_clip_window(
        prompt_emb = prompt_emb,
        ts_emb_map = timestamp_embedding_map,
        k = app_config.clip_duration
    )

    # update the frames in the video to include only frames between start and end
    video["frames"] = {t: img for t, img in video["frames"].items() if start <= t <= end}

    # get reel video
    reel_video_path = render_reel_video(
        video = video,
        output_path = app_config.reel_video_output_folder,
        retain_audio = app_config.retain_audio_in_extracted_clip
    )

    print(f"Reel video rendered - {reel_video_path}")

# tests happen here for now
if __name__ == "__main__":

    main(
        app_config = AppConfig(
            video_path = ".videos/test.mp4",
            prompt = "", # TODO: ChatGPT a prompt
            clip_duration = 10, # lets start with small clips
            reel_video_output_folder = ".output/",
            retain_audio_in_extracted_clip = False # we can toggle this later to play around
        ) 
    )