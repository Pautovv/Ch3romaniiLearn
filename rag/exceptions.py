class RAGError(Exception): pass

class DatasetParsingError(RAGError): pass
class IndexNotFoundError(RAGError): pass
class MetadataNotFoundError(RAGError): pass
class EmptyRetrieverError(RAGError): pass
