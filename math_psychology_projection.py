import example
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "./data"
EMBEDDING_DIM = 30


def split_pdf_document(data_path: str):
    # load pdf documents under DATA_DIR path
    text_loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    loaded_documents = text_loader.load()

    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(loaded_documents)
    print("len chuncked docs: ", len(chunked_documents))
    return chunked_documents


def data_embedd(chunked_documents):
    M_embedd = np.zeros((len(chunked_documents), EMBEDDING_DIM))

    for i in range(0, len(chunked_documents)):
        # getting embedding and adding vector to collection
        embedding = example.get_embedding(chunked_documents[i].page_content)
        M_embedd[i, :] = embedding  # every row is one embedding vector

    return M_embedd


def avarage_embedding(M_embedd):
    avg = M_embedd.mean(0)  # mean row
    return avg


def project_embedding(axis, vector):
    return np.dot(axis, vector)


def create_axis(vec_1, vec_2):
    return vec_1 - vec_2


if __name__ == '__main__':

    ref1 = "./data/cleaned"
    ref2 = "./data"
    projectors = "./data"

    chunked_documents = split_pdf_document(ref1)
    print("chunked documents: ", len(chunked_documents))
    M_embedd = data_embedd(chunked_documents)
    avg_psych = avarage_embedding(M_embedd)
    print(avg_psych)
"""
    chunked_documents = split_pdf_document(ref2)
    M_embedd = data_embedd(chunked_documents)
    avg_math = avarage_embedding(M_embedd)

    axis = create_axis(avg_psych, avg_math)

    chunked_documents = split_pdf_document(projectors)
    M_embedd = data_embedd(chunked_documents)

    # project all vectors and create histogram
    data = np.zeros(len(chunked_documents))

    for i in range(0, len(chunked_documents)):
        z = project_embedding(axis, M_embedd[i, :])
        data[i] = z  # saves all values

    # Plotting a basic histogram
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Basic Histogram')

    # Display the plot
    plt.show()
"""