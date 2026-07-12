import faiss, numpy as np

class FaissStore:
    def __init__(self, d: int) -> None:
        self.index = faiss.IndexFlatIP(d)
    
    def add(self, vector: np.ndarray) -> None:
        self.index.add(vector)
    
    def save(self, filepath: str) -> None:
        faiss.write_index(self.index, filepath)
    
    
    