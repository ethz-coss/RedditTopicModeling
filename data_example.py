from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings

import example
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import chromadb

DATA_DIR = "./data"

hp_collection = example.chroma_client.get_or_create_collection("harry_potter_100")
chunked_documents = []


# splits documents in DATA_DIR and saves them in chunked_documents:[Documents]
def split_pdf_document():
    global chunked_documents
    # load pdf documents under DATA_DIR path
    text_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    loaded_documents = text_loader.load()

    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(loaded_documents)
    print("len chuncked docs: ", len(chunked_documents))


# embeds data in chunked_documents and stores in collection
def run_data_embedd():
    global chunked_documents

    for i in range(2501, len(chunked_documents)):
        # getting embedding and adding vector to collection
        embedding = example.get_embedding(chunked_documents[i].page_content)
        hp_collection.add(
            ids=[str(i)],
            embeddings=[embedding.tolist()],
            documents=[chunked_documents[i].page_content],
        )
        print("added", i)

    print("collection size: ", hp_collection.count())


# queries collection
def query_hp(query_text: str, content: str, n: int):
    #print("collection count: ", hp_collection.count())
    #print("query text: ", query_text)

    # embedding query and searching database
    embedding = example.get_embedding(query_text)
    results = hp_collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=n,
        # where={"metadata_field": "is_equal_to_this"},
        where_document={"$contains": content}
    )

    return results


def reset_hp_database():
    example.chroma_client.delete_collection("harry_potter_100")


# prints id, distance, text
def print_query_results(results):

    ids = results["ids"][0]
    distances = results["distances"][0]
    documents = results["documents"][0]

    for i in range(len(ids)):
        print(ids[i], distances[i], documents[i])

    print("results: ", len(ids))

def get_distance(text1:str, text2:str):
    embedding1 = example.get_embedding(text1)
    embedding2 =example.get_embedding(text2)
    distance = dist = np.linalg.norm(embedding1 - embedding2) #wrong measurement
    return distance


if __name__ == '__main__':
    # prepare data
    # split_pdf_document()
    run_data_embedd()
    #print(get_distance("The dog is brown", "what does the animal look like?"))

    # now we can set a query
    query_text = "Philosopher’s Stone"
    # string that you want to appear in the answers
    contains = " "
    # amount of answers
    n = 3

    results = query_hp(query_text=query_text, content=contains, n=n)
    print_query_results(results=results)  # id, distance, text
