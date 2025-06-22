'''
Audio utility functions
go here
'''
import librosa
import numpy as np

def get_tempo_and_beat_timestamps(audio_path):
    '''
    Loads audio file and returns
    (temp, beat_timestamps)
    '''
    # Load audio and detect beats
    print(f"Processing audio ${audio_path}")

    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_timestamps = librosa.frames_to_time(beat_frames, sr=sr)

    print(f"Detected {len(beat_timestamps)} beats at a tempo of {int(tempo)} BPM")

    return tempo, np.asarray(beat_timestamps)
