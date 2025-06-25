'''
app-wide settings and configs go here
'''

defaults = {
    "model_name": "clip-ViT-B-16", # model being used to generate embeddings for cosine similarity
    "clip_duration": 7, # length of clip to cut from a matching portion of a video
    "max_num_clips": 5, # (max) number of clips to combine out of all matching clips to generate the reel
    "target_width": 720, # output width of video (pixels)
    "target_height": 960, # output height of the video (pixels)
    "match_score_threshold": 0.25, # the minimum threshold for a good match - would change with the model being used
    "retain_audio_in_extracted_clip" : False, # flag to control if the audio of the cut out clips should be retained or disabled
    "frame_sample_rate": 1, # number of frames per second to sample for calculating embeddings
    "audio_sample_rate": 1, # sample rate for applying fourier transform on the audio-wave
    "supported_extensions" : ["mp4"], # supported extensions for video files
 }