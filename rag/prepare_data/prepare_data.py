from datasets import load_dataset

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
    dataset = load_raw_data(dataset_name)

    dataset = dataset.map(parsing_data)

    dataset = dataset.select_columns(['problem', 'solution'])

    return dataset

