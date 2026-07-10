import faiss

class FaissStore:
    def __init__(self, d):
        self.index = faiss.IndexFlatIP(d)
    
    def add(self, vector):
        return self.index.add(vector)
    
    def save(self, filepath):
        return faiss.write_index(self.index, filepath)
    
    
    