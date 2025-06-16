import os

def get_clip_window(prompt_emb, video_path, k):
    frames = get_frames(video_path)
    timestamp_embedding_map = create_timestamp_embedding_map(frames)

    timestamps = timestamp_embedding_map.keys()
    image_embs = timestamp_embedding_map.values()

    clip_window = (0,0)
    n = len(timestamp_embedding_map.items())
    i = 0
    avg_similarity = 0

    while i <= (n-k):
        timestamp_window = timestamps[i:i+k]
        image_emb_window = image_embs[i:i+k]
        image_similarity_window = [cosine_similarity(prompt_emb, image_emb) for image_emb in image_emb_window]

        curr_sum = sum(image_similarity_window)
        curr_avg = curr_sum / k

        if curr_avg > avg_similarity:
            avg_similarity = curr_avg
            i = k
            clip_window = (timestamp_window[0], timestamp_window[len(timestamp_window) - 1])
            
        else:
            i = i + 1

        return clip_window