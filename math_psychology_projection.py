import example
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import matplotlib.pyplot as plt


EMBEDDING_DIM = 5120



def split_pdf_document(data_path: str):
    # load pdf documents under DATA_DIR path
    text_loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    loaded_documents = text_loader.load()
    print("documents loaded")
    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(loaded_documents)
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

def save_to_collection (M_embedd,collection_name:str):
    collection = example.chroma_client.get_or_create_collection(collection_name)
    for i in range(0,len(M_embedd[:,0])):
        collection.add(
            ids=[str(i)],
            embeddings=[M_embedd[i,:].tolist()],
        )

def embeddings_from_collection(collection_name:str):
    #get all vectors from collection and save them in matrix
    collection = example.chroma_client.get_or_create_collection(collection_name)
    embedds = collection.get(
        ids= [str(1)],
        include=["embeddings"]
    )["embeddings"]

    M_embedd = np.zeros((len(embedds),EMBEDDING_DIM))
    
    for i in range(0,len(embedds)):
        M_embedd[i,:]= embedds[i]

    return M_embedd

def new_load_embeddings(source:str, collection_name:str):
    chunked_documents_psych = split_pdf_document(source)
    print("chunked documents: ", len(chunked_documents_psych))
    M_embedd = data_embedd(chunked_documents_psych)
    save_to_collection(M_embedd, collection_name)
    return M_embedd


if __name__ == '__main__':

    ref1 = "./data/psych"
    ref2 = "./data/math"
    data = "./data/math-heavy-articles"
    
    
    #M_embedd = new_load_embeddings(source=ref1, collection_name="ref1")
    M_embedd = embeddings_from_collection("ref1")
    avg_psych = avarage_embedding(M_embedd)
    print("avg: ", avg_psych)
    
    
    #M_embedd = new_load_embeddings(source=ref2, collection_name="ref2")
    M_embedd= embeddings_from_collection("ref2")
    avg_math = avarage_embedding(M_embedd)
    print("avg: ", avg_math)

    axis = create_axis(avg_psych, avg_math)
    print("axis: ", axis)

    #M_embedd = new_load_embeddings(source=data, collection_name="data-math-heavy-articles")
    #M_embedd = embeddings_from_collection("data-mixed-articles")
    M_embedd = embeddings_from_collection("data-math-heavy-articles")
    
    
    # project all vectors and create histogram
    values = np.zeros(len(M_embedd[:,0]))

    print("starting projections")
    for i in range(0, len(M_embedd[:,0])):
        z = project_embedding(axis, M_embedd[i, :])
        values[i] = z  # saves all values

    print("starting histogramm")
    # Plotting a basic histogram
    plt.hist(values, bins=30, color='skyblue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Basic Histogram')

    # Display the plot
    plt.show()
