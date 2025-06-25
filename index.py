'''
This is the main.py file.
You know what its for :)
'''
# python-native library imports
import argparse

# third-party libraries import
import dotenv

# modules import
from modules.vector import load_model, compute_embeddings
from modules.video import get_top_match_clips,get_concatenated_video, sync_cuts_to_nearest_beat, render_reel
from modules.audio import get_tempo_and_beat_timestamps, load_audio_clip
from modules.file import pre_flight_checks,get_input_videos_list

# configs import
from configs.settings import defaults

# load env variables
dotenv.load_dotenv()

# load model - outside all loops for efficiency
MODEL_NAME= "clip-ViT-B-16"
model = load_model(MODEL_NAME)


def generate_reel(input_folder, output_folder, scene_prompt, audio_path):
    '''
    main method - you know what it does :)
    '''
    # perform pre-flight checks
    pre_flight_checks(
        input_folder = input_folder,
        output_folder = output_folder
    )

    # get list of videos in specified folder
    video_list = get_input_videos_list(
        input_folder = input_folder
    )

    if len(video_list) > 0:

        print(f"{len(video_list)} supported videos in {input_folder}")

        # compute embeddings for the prompt
        print("Computing embeddings for the scene prompt...")
        prompt_emb = compute_embeddings(
            item = scene_prompt,
            model = model
        )

        # get top-match clips from video-list
        print("Getting top matches from the list of videos...")
        top_match_clips = get_top_match_clips(
            video_list = video_list,
            prompt_emb = prompt_emb,
            model = model
        )

        # get initial concatenated video
        concatenated_video = get_concatenated_video(
            video_segments = top_match_clips,
            retain_audio = defaults['retain_audio_in_extracted_clip']
        )

        # load audio file, extract tempo and beat_timestamps
        tempo, beat_timestamps = get_tempo_and_beat_timestamps(audio_path=audio_path)

        # each cut should have same tempo as parent audio - useful for syncing cuts to beats
        cut_tempo = int((tempo * (defaults['clip_duration'] - 1)) / 60)

        # get video segments with cuts synced to the beats in the audio
        synced_video = sync_cuts_to_nearest_beat(
            video = concatenated_video,
            cut_tempo = cut_tempo,
            beat_timestamps = beat_timestamps
        )
        
        # load the final audio
        audio = load_audio_clip(
            audio_path = audio_path,
            beat_timestamps = beat_timestamps,
            duration = synced_video.duration
        )

        # render the final reel 
        render_reel(
            final_video = synced_video,
            final_audio = audio,
            output_folder = output_folder
        )
    else:
        print(f"No supported video files found in {input_folder}")

if __name__ == "__main__":

    # parse CLI arguments
    parser = argparse.ArgumentParser(description="AI Reel Generator")
    parser.add_argument(
        '-i', '--input-folder',
        type=str,
        required=True, 
        help="Folder containing the source videos - absolute path"
    )
    parser.add_argument(
        '-o','--output-folder',
        type=str, 
        help="Folder to render the reel in - absolute path"
    )
    parser.add_argument(
        '-s','--scene-prompt',
        type=str,
        required=True,
        help="The prompt for extracting scenes from videos"
    )
    parser.add_argument(
        '-a','--audio-path',
        type=str,
        required=True,
        help="The music to be added to the reel"
    )
    args = parser.parse_args()

    # generate reel using the given args
    generate_reel(
        input_folder = args.input_folder,
        output_folder = args.output_folder,
        scene_prompt = args.scene_prompt,
        audio_path = args.audio_path
    )