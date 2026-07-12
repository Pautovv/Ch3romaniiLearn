import faiss, json, logging
from rag.embedding_model.embedding_model import load_model, encode

from rag.config import MODEL_NAME, INDEX_PATH, METADATA_PATH

logger = logging.getLogger(__name__)

model = load_model(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f) 

def retriever(user_query, top_k):
    logger.info(f'Search started | Query length: {len(user_query)} | Top_K: {top_k}')

    query_embedding = encode([user_query], model)
    faiss.normalize_L2(query_embedding)

    scores, ids = index.search(query_embedding, top_k)
    
    documents = [metadata[idx] for idx in ids[0]]

    if documents:
        top_score = float(scores[0][0])
        logger.info(f'Found {len(documents)} documents | Top score: {top_score:.4f}')
    else:
        logger.warning('No documents found for the query')
        
    return documents

    
