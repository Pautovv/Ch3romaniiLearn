import logging
from datasets import load_dataset

from datasets.exceptions import DatasetNotFoundError
from requests.exceptions import ConnectionError
from rag.exceptions import DatasetParsingError

logger = logging.getLogger(__name__)

def load_raw_data(dataset_name):
    try:
        dataset = load_dataset(dataset_name)
    except DatasetNotFoundError as e:
        logger.error(f'Dataset {dataset_name} not found on HuggingFace hub')
        raise DatasetParsingError(
            f'Датасет {dataset_name} не был найден на HF Hub: проверте название.'
        ) from e
    except ConnectionError as e:
        logger.error(f'Network error while downloading {dataset_name}')
        raise DatasetParsingError(
            f'Не удалсоь скачать датасет: проблемы с сетью.'
        ) from e
    try:
        return dataset['train']
    except KeyError as e:
        logger.error('Dataset has not train split')
        raise DatasetParsingError(
            f'Не удалось вернуть сплит датасета: проверте существования train-сплита.'
        ) from e

def parsing_data(dataset):
    problem = dataset['question']
    solution = dataset['ground_truth']

    return {
        'problem' : problem,
        'solution' : solution
    }

def prepare_dataset(dataset_name):
    logger.info(f'Starting data preparing for {dataset_name}...')
    dataset = load_raw_data(dataset_name)
    logger.info(f'Downloaded {dataset.shape[0]:,} rows')

    dataset = dataset.map(parsing_data)
    logger.info('Applied parsing data')

    dataset = dataset.select_columns(['problem', 'solution'])
    logger.info(f'Prepared {dataset.shape[0]:,} rows and {list(dataset.features.keys())} cols')

    return dataset

