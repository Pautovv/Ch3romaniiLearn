import logging, requests

from app.config import OLLAMA_MODEL, OLLAMA_URL

from requests.exceptions import ConnectionError
from app.exceptions import ModelLoadingError, EmptyGeneratorError

logger = logging.getLogger(__name__)

class OllamaGenerator:
    def __init__(self, model_name: str = OLLAMA_MODEL) -> None:
        self.model_name = model_name
  
    def generate(self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float) -> str:
        logger.info('Generating response...')

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    'model': self.model_name,
                    'messages': messages,
                    'stream': False,
                    'options' : {
                        'temperature': temperature,
                        'num_predict': max_new_tokens
                    }
                }
            )
        except ConnectionError as e:
            logger.error('Ollama server is not reachable')
            raise ModelLoadingError(
                'Не удалось подключиться к Ollama-серверу: проверте его запуск.'
            ) from e
        
        if response.status_code != 200:
            logger.error(f'Ollama returned status code: {response.status_code}: {response.text}')
            raise ModelLoadingError(f'Ollama вернул ошибку: {response.text}.')
        
        res = response.json()['message']['content']

        if res:
            logger.info('Response generated')
            return res
        else:
            logger.warning('Response is empty')
            raise EmptyGeneratorError('Сгенерированный ответ модели пустой.')