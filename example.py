import numpy as np
import requests
import json
import chromadb

chroma_client = chromadb.HttpClient(host='localhost', port=8000)


def get_embedding(sentence: str) -> np.ndarray:
    url = 'http://localhost:5000/v1/embeddings'

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    data = {
        "input": sentence
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        raise ValueError(f'Error: {response.status_code}')
    else:
        embedding_vector = np.array(response.json()['data'][0]['embedding'])
        return embedding_vector


def insert(vid: str, sentence: str, embedding: np.ndarray) -> None:
    collection = chroma_client.get_or_create_collection(name="my_collection")
    print("my_collection count: ", collection.count())
    collection.add(
        ids=[vid],
        embeddings=[embedding.tolist()],
        documents=[sentence],
    )
    print(collection.count())

def peek():
    print(chroma_client.get_collection(name="my_collection").peek())


def run_example():
    sentence = 'Here is an example sentence'
    embedding = get_embedding(sentence=sentence)
    insert(vid='id1', sentence=sentence, embedding=embedding)
    peek()


if __name__ == '__main__':
    #run_example()
    print(chroma_client.list_collections())
    
     
