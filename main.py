'''
This is the main.py file.
You know what its for :)
'''
import json
import time
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from utils.video_utils import extract_video_info_and_frames,render_reel_video
from utils.vector_utils import compute_embeddings, get_clip_window
from utils.file_utils import pre_flight_checks,get_input_videos_list
from config import AppConfig

# load model - outside all loops for efficiency
MODEL_NAME = "clip-ViT-B-32"
print(f"Loading model {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)

def main(app_config):
    '''
    main method - you know what it does :)
    '''
    # perform pre-flight checks
    pre_flight_checks(app_config=app_config)

    # compute embeddings for the prompt
    print("Computing embeddings for the scene prompt...")
    prompt_emb = compute_embeddings(
        item = app_config.scene_prompt,
        model = model
    )

    # get list of videos in specified folder
    video_list = get_input_videos_list(input_folder=app_config.video_input_folder)
    if len(video_list) == 0:
        print(f"No supported video files in {app_config.video_input_folder}")
    else:
        print(f"Found {len(video_list)} supported files in input folder")

        for video_path in video_list:

            # extract video info and frames
            video = extract_video_info_and_frames(
                video_path = video_path,
                sampling_rate = app_config.frame_sampling_rate
            )

            # create a map of timestamps -> image embeddings for each video frame
            print("Computing embeddings for the sampled frames ...")
            timestamp_embedding_map = {t: compute_embeddings(item=img["pil_image"], model=model) for t, img in tqdm(video["frames"].items())}
        
            # get start and end times for the clip to cut out of the video
            print("Extracting clip with best match score ...")
            start, end = get_clip_window(
                prompt_emb = prompt_emb,
                ts_emb_map = timestamp_embedding_map,
                k = app_config.clip_duration,
                threshold = app_config.match_score_threshold
            )

            # get reel video
            video_file_name = video_path.split("/")[-1:][0].split(".")[0]
            output_path = f"{app_config.video_output_folder}/{video_file_name}_reel_{time.time()}.mp4"
            render_reel_video(
                video_path=video_path,
                output_path=output_path,
                start=start,
                end=end,
                retain_audio=app_config.retain_audio_in_extracted_clip
            )

            print("Reel clip rendered!")

# tests happen here for now
if __name__ == "__main__":

    with open("test_cases.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test case(s) from test_cases.json")

    for test_case in test_cases :

        main(
            app_config = AppConfig(
                video_input_folder = test_case["video_input_folder"],
                scene_prompt = test_case["scene_prompt"],
                clip_duration = test_case["clip_duration"],
                video_output_folder = test_case["video_output_folder"],
                retain_audio_in_extracted_clip = test_case["retain_audio_in_extracted_clip"],
                frame_sampling_rate = test_case["frame_sampling_rate"],
                match_score_threshold = test_case["match_score_threshold"],
                tags = test_case["tags"],
            )
       )