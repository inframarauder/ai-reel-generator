'''
    utility methods for all vector/AI operations like 
    computing embeddings and cosine similarity
'''
from sentence_transformers import util

def compute_embeddings(item,model):
    '''
        Returns embeddings using from a 
        given model_name
    '''
    return model.encode(item)


def get_clip_window(prompt_emb, ts_emb_map, k):
    '''
        method to get a clip window (start,end) of size k
        where the frames have the highest similarity to the prompt

        uses the sliding-window algorithm
    '''
    timestamps = list(ts_emb_map.keys())
    image_embs = list(ts_emb_map.values())

    clip_window = (0,0)
    n = len(ts_emb_map.items())
    i = 0
    max_avg_similarity = 0

    while i <= (n-k):
        timestamp_window = timestamps[i:i+k]
        image_emb_window = image_embs[i:i+k]
        image_similarity_window = [util.cos_sim(prompt_emb, image_emb).flatten()[0] for image_emb in image_emb_window]

        curr_sum = sum(image_similarity_window)
        curr_avg = curr_sum / k

        if curr_avg > max_avg_similarity:
            max_avg_similarity = curr_avg
            i = k
            clip_window = (timestamp_window[0], timestamp_window[len(timestamp_window) - 1])
            
        else:
            i = i + 1

    print(f"Max match score: {max_avg_similarity}\nClip Cut Points (sec): {clip_window}")
    return clip_window