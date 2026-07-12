import logging
from sentence_transformers import SentenceTransformer

from huggingface_hub.errors import RepositoryNotFoundError
from requests.exceptions import ConnectionError
from rag.exceptions import ModelLoadingError

logger = logging.getLogger(__name__)

def load_model(model_name):
    try:
        return  SentenceTransformer(model_name)
    except RepositoryNotFoundError as e:
        logger.error(f'Model {model_name} not found on HuggingFace Hub')
        raise ModelLoadingError(
            f'Модель {model_name} не была найдена на HF Hub: проверьте название.'
        ) from e 
    except ConnectionError as e:
        logger.error(f'Network error while downloading {model_name}')
        raise ModelLoadingError(
            f'Не удалсоь скачать модель: проблемы с сетью.'
        ) from e
    
def encode(sentences, model):
    embeddings = model.encode(sentences)
    
    return embeddings





