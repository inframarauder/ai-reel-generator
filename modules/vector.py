'''
utility methods for all vector/AI operations like 
computing embeddings and cosine similarity
'''
from sentence_transformers import SentenceTransformer,util

def load_model(model_name):
    '''
    Loads and returns the model
    using SentenceTransformer
    '''
    print(f"Loading {model_name} ...")
    model = SentenceTransformer(model_name)
    
    return model

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


