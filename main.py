'''
This is the main.py file.
You know what its for :)
'''
import json
import heapq
from tqdm import tqdm
import dotenv

from sentence_transformers import SentenceTransformer
from utils.video_utils import extract_video_info_and_frames,render_reel
from utils.vector_utils import compute_embeddings, get_clip_window_match_score
from utils.file_utils import pre_flight_checks,get_input_videos_list
from config import AppConfig

# load env variables
dotenv.load_dotenv()

# load model - outside all loops for efficiency
MODEL_NAME = "clip-ViT-L-14"
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
        shortlisted_clips = []
        heapq.heapify(shortlisted_clips)

        for video_path in video_list:

            # extract video info and frames
            video = extract_video_info_and_frames(
                video_path = video_path,
                sampling_rate = app_config.frame_sampling_rate
            )

            if video["total_duration"] < app_config.clip_duration:
                print(f"Ignoring clip {video_path} as duration is too small..")
            else:
                # create a map of timestamps -> image embeddings for each video frame
                print("Computing embeddings for the sampled frames ...")
                timestamp_embedding_map = {t: compute_embeddings(item=img["pil_image"], model=model) for t, img in tqdm(video["frames"].items())}
            
                # get start and end times for the clip to cut out of the video
                print("Extracting clip with best match score ...")
                (clip_window, match_score) = get_clip_window_match_score(
                    prompt_emb = prompt_emb,
                    ts_emb_map = timestamp_embedding_map,
                    k = app_config.clip_duration,
                    threshold = app_config.match_score_threshold
                )
                
                # only clips with match_score over the threshold
                if match_score > app_config.match_score_threshold:
                    heapq.heappush(shortlisted_clips, (match_score, clip_window,video_path))

        top_match_clips = heapq.nlargest(app_config.num_clips,shortlisted_clips)

        render_reel(
            video_segments = top_match_clips,
            output_folder = app_config.video_output_folder,
            retain_audio = app_config.retain_audio_in_extracted_clip
        )
        

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
                num_clips=test_case["num_clips"],
                video_output_folder = test_case["video_output_folder"],
                retain_audio_in_extracted_clip = test_case["retain_audio_in_extracted_clip"],
                frame_sampling_rate = test_case["frame_sampling_rate"],
                match_score_threshold = test_case["match_score_threshold"],
                tags = test_case["tags"],
            )
       )