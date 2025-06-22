'''
App-wide configurations go here
'''
from dataclasses import dataclass

@dataclass
class AppConfig:
    '''
    All the attributes that can be configured for the app
    '''
    video_input_folder: str
    scene_prompt: str
    clip_duration: int
    num_clips: int
    video_output_folder: str
    retain_audio_in_extracted_clip: bool
    frame_sampling_rate: int
    match_score_threshold: float
    tags: list[str]