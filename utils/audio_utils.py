'''
Audio utility functions
go here
'''
import librosa

def get_tempo_and_beat_timestamps(audio_path):
    '''
    Loads audio file and returns
    (temp, beat_timestamps)
    '''
    # Load audio and detect beats
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_timestamps = librosa.frames_to_time(beat_frames, sr=sr)

    return tempo, beat_timestamps


def find_optimal_beat_segment(tempo, beat_timestamps, target_duration):
    """
    Finds the best audio segment that:
    1. Starts and ends on beats
    2. Matches target duration as closely as possible
    """
    # Calculate req number of beats
    req_beat_count = round(target_duration * tempo / 60)
    
    # Sliding window to find best segment
    optimal_segment = tuple()
    min_duration_diff = float('inf')
    n = len(beat_timestamps)
    k = req_beat_count
    
    for i in range(n - k + 1):
        start = beat_timestamps[i]
        end = beat_timestamps[i + k - 1]
        duration = end - start
        duration_diff = abs(duration - target_duration)
        
        # Prefer segments closer to target duration
        if duration_diff < min_duration_diff:
            min_duration_diff = duration_diff
            optimal_segment = (start, end)
        
        # Early exit if we find a perfect match
        if duration_diff == 0:
            break
    
    if not optimal_segment:
        raise ValueError("Could not find suitable beat-aligned segment")
    
    return optimal_segment
