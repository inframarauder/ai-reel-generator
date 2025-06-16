
'''
    App-wide configurations go here
'''
from dataclasses import dataclass

@dataclass
class AppConfig:
    '''
        All the attributes that can be configured for the app
    '''
    video_path: str
    prompt: str
    clip_duration: int
    reel_video_output_folder: str
    retain_audio_in_extracted_clip: bool