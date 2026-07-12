import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)

def load_raw_data(dataset_name):
    dataset = load_dataset(dataset_name)

    return dataset['train']

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

