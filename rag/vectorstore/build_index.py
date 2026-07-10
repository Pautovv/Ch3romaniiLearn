import json, faiss
from rag.prepare_data.prepare_data import prepare_dataset
from rag.embedding_model.embedding_model import load_model, encode
from rag.vectorstore.faiss_store import FaissStore

def build_index(dataset_name, model_name, filepath):
    dataset = prepare_dataset(dataset_name)
    model = load_model(model_name)

    dataset = dataset.map(
        lambda raw: {'document' : f"{raw['problem']}\n\n{raw['solution']}"}
    )

    embeddings = encode(dataset['document'], model)
    d = embeddings.shape[1]

    index = FaissStore(d)

    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    metadata = list(dataset['document'])

    index.save(filepath)
    with open('rag/data/processed/metadata.json', 'w') as f:
        json.dump(metadata, f)

build_index('eth-nlped/mathdial', 'sentence-transformers/all-MiniLM-L6-v2', 'rag/data/processed/index.faiss')


