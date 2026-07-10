from sentence_transformers import SentenceTransformer

def load_model(model_name):
    model = SentenceTransformer(model_name)

    return model

def encode(sentences, model):
    embeddings = model.encode(sentences)
    
    return embeddings





