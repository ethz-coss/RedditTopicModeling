import numpy as np

import example

collection2 = example.chroma_client.get_or_create_collection("random_sentence")


def insert(vid: str, sentence: str, embedding: np.ndarray) -> None:
    global collection2

    print("my_collection count: ", collection2.count())
    collection2.add(
        ids=[vid],
        embeddings=[embedding.tolist()],
        documents=[sentence],
    )
    print(collection2.count())


def query(query_text: str, content: str, n: int):

    print("collection count: ", collection2.count())
    print("query text: ", query_text)

    # embedding query and searching database
    embedding = example.get_embedding(query_text)
    results = collection2.query(
        query_embeddings=[embedding.tolist()],
        n_results=n,
        # where={"metadata_field": "is_equal_to_this"},
        where_document={"$contains": content}
    )

    print_query_results(results=results)


def reset_database():
    example.chroma_client.delete_collection("random_sentence")


# prints id, distance, text
def print_query_results(results):
    #print("results: ", results)
    ids = results["ids"][0]
    distances = results["distances"][0]
    documents = results["documents"][0]
    for i in range(len(ids)):
        print(ids[i], distances[i], documents[i])


if __name__ == '__main__':
    #reset_database()
    collection2 = example.chroma_client.get_or_create_collection("random_sentence")

    #adding 4 sentences
    s1 = 'The dog is brown'
    e1 = example.get_embedding(sentence=s1)
    insert(vid='1', sentence=s1, embedding=e1)

    s2 = 'I love the sky'
    e2 = example.get_embedding(sentence=s2)
    insert(vid='2', sentence=s2, embedding=e2)

    s3 = 'She stood at the traffic light'
    e3 = example.get_embedding(sentence=s3)
    insert(vid='3', sentence=s3, embedding=e3)

    s4 = 'Peaches '
    e4 = example.get_embedding(sentence=s4)
    insert(vid='4', sentence=s4, embedding=e4)

    #queries
    query_text = "Peaches "
    contains = " "
    n = 3
    query(query_text=query_text, content=contains, n=n)

    query_text = "What is the gender of the person waiting?"
    contains = " "
    n = 4
    query(query_text=query_text, content=contains, n=n)

    query_text = "why is she standing around?"
    contains = " "
    n = 4
    query(query_text=query_text, content=contains, n=n)

    query_text = "Do I like looking up?"
    contains = " "
    n = 3
    query(query_text=query_text, content=contains, n=n)

    query_text = "How do I feel about clouds?"
    contains = " "
    n = 3
    query(query_text=query_text, content=contains, n=n)

    query_text = ("animal")
    contains = " "
    n = 3
    query(query_text=query_text, content=contains, n=n)

    query_text = "what does the animal look like?"
    contains = " "
    n = 3
    query(query_text=query_text, content=contains, n=n)

    query_text = "what do I like?"
    contains = " "
    n = 3
    query(query_text=query_text, content=contains, n=n)

    query_text = "what do I love?"
    contains = " "
    n = 3
    query(query_text=query_text, content=contains, n=n)

    query_text = "what is the color of the dog?"
    contains = " "
    n = 3
    query(query_text=query_text, content=contains, n=n)

