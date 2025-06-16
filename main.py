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
from config import AppConfig

# load model - outside loop for efficiency
MODEL_NAME = "clip-ViT-B-32"
print(f"Loading model {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)

def main(app_config):
    '''
        main method - you know what it does :)
    '''
    print(f"\nProcessing video: {app_config.video_path}")

    # compute embeddings for the prompt
    print("Computing embeddings for the prompt...")
    prompt_emb = compute_embeddings(
        item = app_config.prompt,
        model = model
    )

    # extract video info and frames
    video = extract_video_info_and_frames(
        video_path = app_config.video_path,
        sampling_rate = app_config.sampling_rate
    )

    # create a map of timestamps -> image embeddings for each video frame
    print("Computing embeddings for the sampled frames ...")
    timestamp_embedding_map = {t: compute_embeddings(item=img["pil_image"], model=model) for t, img in tqdm(video["frames"].items())}
   
    # get start and end times for the clip to cut out of the video
    print("Extracting clip with best match score ...")
    start, end = get_clip_window(
        prompt_emb = prompt_emb,
        ts_emb_map = timestamp_embedding_map,
        k = app_config.clip_duration
    )

    # get reel video
    output_path = f"{app_config.reel_video_output_folder}/reel_{time.time()}.mp4"
    render_reel_video(
        video_path=app_config.video_path,
        output_path=output_path,
        start=start,
        end=end,
        retain_audio=app_config.retain_audio_in_extracted_clip
    )

    print(f"Reel video rendered!")

# tests happen here for now
if __name__ == "__main__":

    with open("test_cases.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test case(s) from test_cases.json")

    for test_case in test_cases :

        main(
            app_config = AppConfig(
                video_path = test_case["video_path"],
                prompt = test_case["prompt"],
                clip_duration = test_case["clip_duration"],
                reel_video_output_folder = test_case["reel_video_output_folder"],
                retain_audio_in_extracted_clip = test_case["retain_audio_in_extracted_clip"],
                sampling_rate = test_case["sampling_rate"]
            )
       )