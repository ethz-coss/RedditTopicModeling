from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings
import time
import example
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import tensorflow as tf

DATA_DIR = "./data/psych"
EMBEDDING_DIM = 5120
hp_collection = example.chroma_client.get_or_create_collection("harry_potter_100")
chunked_documents : list[str]


# splits documents in DATA_DIR and saves them in chunked_documents:[Documents]
def split_pdf_document():
    global chunked_documents
    # load pdf documents under DATA_DIR path
    text_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    loaded_documents = text_loader.load()

    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0)
    chunked_document = text_splitter.split_documents(loaded_documents)
    chunked_documents = [chunked_document[i].page_content for i in range(len(chunked_document))]
    print("len chuncked docs: ", len(chunked_documents))


# embeds data in chunked_documents and stores in collection
def run_data_embedd():
    global chunked_documents

    for i in range(0, len(chunked_documents)):
        # getting embedding and adding vector to collection
        embedding = example.get_embedding(chunked_documents[i])
        hp_collection.add(
            ids=[str(i)],
            embeddings=[embedding.tolist()],
            documents=[chunked_documents[i]],
        )
        print("added", i)

    print("collection size: ", hp_collection.count())


def run_data_embedd_fast():
    
    #convert text dataset into Tensorflow Dataset
    dataset = tf.data.Dataset.from_tensor_slices(chunked_documents)
    embeddings_dataset = dataset.map(example.get_embedding2)
    
    for i in range(0, len(chunked_documents)):
        # getting embedding and adding vector to collection
        embedding = next(iter(embeddings_dataset))
        hp_collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunked_documents[i]],
        )
        print("added", i)

    print("collection size: ", hp_collection.count())

def save_to_collection(M_embedd):
    for i in range(0, len(M_embedd[:, 0])):
        hp_collection.add(
            ids=[str(i)],
            embeddings=[M_embedd[i, :].tolist()],
        )


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
    split_pdf_document()

    start = time.time()
    M = run_data_embedd_fast()
    save_to_collection(M)
    print('time: ', (time.time()-start))

    start = time.time()
    run_data_embedd()
    print('time: ', (time.time()-start))
    #print(get_distance("The dog is brown", "what does the animal look like?"))

