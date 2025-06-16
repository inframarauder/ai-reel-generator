'''
    utility methods for all vector/AI operations like 
    computing embeddings and cosine similarity
'''

# TODO: import all AI dependencies like torch, numpy, sentence-transformer, etc..


def compute_prompt_embedding(prompt):
    '''
        method to take a text prompt as an input and
        return the embeddings computed for it
    '''
    # TODO: implement method
    return []

def compute_image_embedding(image):
    '''
        method to take an image as an input and
        return the embeddings computed for it
    '''
    # TODO: implement method
    return []


def get_clip_window(prompt_emb, ts_emb_map, k):
    '''
        method to get a clip window (start,end) of size k
        where the frames have the highest similarity to the prompt

        uses the sliding-window algorithm
    '''
    timestamps = ts_emb_map.keys()
    image_embs = ts_emb_map.values()

    clip_window = (0,0)
    n = len(ts_emb_map.items())
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