import logging
from app.generation.prompt_builder import build_prompt

from app.generation.inference import Generator
from rag.retriever.retriever import Retriever

from app.config import TOP_K, MAX_NEW_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)

def generate_answer(
        user_query: str, 
        retriever: Retriever, 
        generator: Generator, 
        top_k: int = TOP_K, 
        max_new_tokens: int = MAX_NEW_TOKENS, 
        temperature: float = TEMPERATURE
    ) -> str:
    logger.info('Pipeline started')
    documents = retriever.retrieve(user_query, top_k)
    messages = build_prompt(user_query, documents)

    response = generator.generate(messages, max_new_tokens, temperature)
    logger.info('Pipeline completed')
    return response

