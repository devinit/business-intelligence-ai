import os
import pickle
from openai import OpenAI
from dotenv import load_dotenv
from embedding_retrieval import load_embeddings, run_query

load_dotenv()
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)


MODEL = "gpt-4o-mini"

SYSTEM_PROMPT_TEMPLATE = """
You are Development Initiative's (DI) Business Intelligence AI, responsible for helping answer employee questions about institutional knowledge.
The context of internal documents that match the user's query are below, you will use this context to answer their question.
Context start:
{}
Context end.
If the content states 'I don't know' you must also tell the user you do not know the answer to the question, as it may be outside of the scope of your task.
"""

CACHE_PATH = "large_data/humanitarian_cache.pkl"


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as pickle_file:
            cache = pickle.load(pickle_file)
    else:
        cache = dict()
    return cache


def save_cache(cache):
    with open(CACHE_PATH, 'wb') as pickle_file:
        pickle.dump(cache, pickle_file)


if __name__ == '__main__':
    data_source = "data/humanitarian_acronyms"
    full_text_list, file_embeddings = load_embeddings(data_source)
    cache = load_cache()
    while True:
        query = input("> ")
        results = run_query(full_text_list, file_embeddings, query)
        context = "\n".join(results)
        
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context)
        user_prompt = query

        if query in cache:
            print("Cached response: {}".format(cache[query]))
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            answer = response.choices[0].message.content
            cache[query] = answer
            save_cache(cache)
            print(answer)
