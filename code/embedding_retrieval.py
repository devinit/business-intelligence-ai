import os
from glob import glob
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()


global MODEL
global DEVICE
MODEL = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-long", trust_remote_code=True)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = MODEL.to(DEVICE)


def load_embeddings(data_source):
    data_source_basename = os.path.basename(data_source)
    pickle_path = os.path.join("large_data", "{}.pkl".format(data_source_basename))
    input_dir = os.path.abspath(data_source)
    txt_wildcard = os.path.join(input_dir, '*.txt')
    txt_files = glob(txt_wildcard)

    full_text_list = list()
    for txt_file_path in tqdm(txt_files):
        with open(txt_file_path, 'r') as txt_file:
            full_text_list.append(txt_file.read())

    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as pickle_file:
            file_embeddings = pickle.load(pickle_file)
    else:
        file_embeddings = list()
        for full_text in tqdm(full_text_list):
            embedding = MODEL.encode(full_text)
            file_embeddings.append(embedding)
        with open(pickle_path, 'wb') as pickle_file:
            pickle.dump(file_embeddings, pickle_file)
    return full_text_list, file_embeddings


def run_query(full_text_list, file_embeddings, query, threshold=0.35, debug=False):
    query_embedding = MODEL.encode(query, prompt_name="query")
    ranks = np.zeros(len(file_embeddings))
    for i, embedding in enumerate(file_embeddings):
        ranks[i] = query_embedding @ embedding.T

    dataset = Dataset.from_dict(
        {
            'text': full_text_list,
            'rank': ranks
        }
    )

    matching_dataset = dataset.filter(lambda example: example['rank'] > threshold)
    if matching_dataset.num_rows == 0:
        return ["I don't know."]
    else:
        sorted_matching_dataset = matching_dataset.sort('rank', reverse=True)
        if debug:
            return sorted_matching_dataset['rank'], sorted_matching_dataset['text']
        return sorted_matching_dataset['text']


def main(data_source):
    full_text_list, file_embeddings = load_embeddings(data_source)
    while True:
        query = input("> ")
        results = run_query(full_text_list, file_embeddings, query)
        for result in results:
            print(result)


if __name__ == '__main__':
    data_source = "data/humanitarian_acronyms"
    main(data_source)