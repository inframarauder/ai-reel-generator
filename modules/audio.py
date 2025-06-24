'''
Audio utility functions
go here
'''
import librosa
import numpy as np
from moviepy import AudioFileClip
from moviepy.audio.fx import AudioFadeIn,AudioFadeOut

def get_tempo_and_beat_timestamps(audio_path):
    '''
    Loads audio file and returns
    (temp, beat_timestamps)
    '''
    # Load audio and detect beats
    print(f"Processing audio {audio_path}")

    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_timestamps = librosa.frames_to_time(beat_frames, sr=sr)

    print(f"Detected {len(beat_timestamps)} beats at a tempo of {int(tempo)} BPM")

    return tempo.flatten()[0], np.asarray(beat_timestamps) # type: ignore

def load_audio_clip(audio_path, beat_timestamps, duration):
    '''
    Load audio clip from file 
    and return subclip based on beat_timestamp and duration
    '''
    start = beat_timestamps[0]
    end = duration + start

    audio_clip = AudioFileClip(audio_path).subclipped(start, end)

    audio_clip = AudioFadeIn(1).apply(audio_clip)
    audio_clip = AudioFadeOut(2).apply(audio_clip)

    return audio_clip
