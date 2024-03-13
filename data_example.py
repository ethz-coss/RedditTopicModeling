from langchain_community.vectorstores.chroma import Chroma

import example
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

DATA_DIR = "./data"

collection = example.chroma_client.get_or_create_collection("harry_potter_100")
chunked_documents = []


# splits documents in Data_Dir and saves them in chuncked_documents:[Documents]
def split_pdf_document():
    global chunked_documents
    # load pdf documents under DATA_DIR path
    text_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    loaded_documents = text_loader.load()

    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(loaded_documents)
    print("len chuncked docs: ", len(chunked_documents))


# embedds data in chuncked_documents and stores in collection
def run_data_embedd():
    global chunked_documents

    for i in range(1596, len(chunked_documents)):
        # getting embedding and adding vector to collection
        embedding = example.get_embedding(chunked_documents[i].page_content)
        collection.add(
            ids=[str(i)],
            embeddings=[embedding.tolist()],
            documents=[chunked_documents[i].page_content],
        )
        print("added", i)

    print("collection size: ", collection.count())


# queries collection
def query(query_text: str, content: str, n: int):
    global chunked_documents
    print("collection count: ", collection.count())
    print("query text: ", query_text)

    # embedding query and searching database
    embedding = example.get_embedding(query_text)
    results = collection.query(
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
    # print("results: ", results)
    ids = results["ids"][0]
    distances = results["distances"][0]
    for i in range(len(ids)):
        print(ids[i], distances[i], chunked_documents[int(ids[i])].page_content)


if __name__ == '__main__':
    # prepare data
    print(collection.count())
    split_pdf_document()
    # run_data_embedd()

    # now we can set a query
    query_text = "where is Hogwarts?"
    # string that you want to appear in the answers
    contains = " "
    # amount of answers
    n = 10

    results = query(query_text=query_text, content=contains, n=n)
    print_query_results(results=results)  # id, distance, text
