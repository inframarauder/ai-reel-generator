'''
utility methods for all vector/AI operations like 
computing embeddings and cosine similarity
'''
from sentence_transformers import util

def cosine_similarity(vector1,vector2):
    '''
    returns cosine similarity between two 1-D vectors
    '''
    return util.cos_sim(vector1, vector2).flatten()[0]

def compute_embeddings(item,model):
    '''
    Returns embeddings using from a 
    given model_name
    '''
    return model.encode(item)


def get_clip_window_match_score(prompt_emb, ts_emb_map, k, threshold):
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
    max_avg_similarity = threshold

    while i <= (n-k):
        timestamp_window = timestamps[i:i+k]
        image_emb_window = image_embs[i:i+k]
        image_similarity_window = [cosine_similarity(prompt_emb,image_emb) for image_emb in image_emb_window]

        curr_sum = sum(image_similarity_window)
        curr_avg = round(float(curr_sum / k), 5)
        
        # find max avg
        if curr_avg > max_avg_similarity:
            max_avg_similarity = curr_avg
            i = k
            clip_window = (timestamp_window[0], timestamp_window[len(timestamp_window) - 1])
            
        else:
            i = i + 1

    return (clip_window, max_avg_similarity)

