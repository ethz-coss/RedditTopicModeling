import example
import data_splitter


import numpy as np
import matplotlib.pyplot as plt


EMBEDDING_DIM = 5120


def data_embedd(chunked_documents):
    M_embedd = np.zeros((len(chunked_documents), EMBEDDING_DIM))

    for i in range(0, len(chunked_documents)):
        # getting embedding and adding vector to collection
        embedding = example.get_embedding(chunked_documents[i].page_content)
        M_embedd[i, :] = embedding  # every row is one embedding vector

    return M_embedd

def data_embedd_n(chunked_documents):
    M_embedd = np.zeros((len(chunked_documents), EMBEDDING_DIM))

    for i in range(0, len(chunked_documents)):
        # getting embedding and adding vector to collection
        embedding = example.get_embedding(chunked_documents[i].page_content)

        embedding = embedding / np.linalg.norm(embedding)

        M_embedd[i, :] = embedding  # every row is one embedding vector

    return M_embedd


def avarage_embedding(M_embedd):
    avg = M_embedd.mean(0)  # mean row
    return avg

def avarage_embedding_n(M_embedd):
    avg = M_embedd.mean(0)  # mean row
    avg = avg / np.linalg.norm(avg) 
    return avg


def create_axis(vec_1, vec_2):
    return vec_1 - vec_2

def create_axis_n(vec_1, vec_2):
    x = vec_1 - vec_2
    return x / np.linalg.norm(x)


def new_load_embeddings(source:str, collection_name:str):
    chunked_documents = data_splitter.split_pdf_document(source)
    print("chunked documents: ", len(chunked_documents))
    M_embedd = data_embedd(chunked_documents)

    meta = [chunked_documents[i].metadata for i in range(0, len(chunked_documents))]

    save_to_collection(M_embedd, meta, collection_name)
    return M_embedd, meta

def new_load_embeddings_n(source:str, collection_name:str):
    chunked_documents = data_splitter.split_pdf_document(source)
    print("chunked documents: ", len(chunked_documents))
    M_embedd = data_embedd_n(chunked_documents)

    meta = [chunked_documents[i].metadata for i in range(0, len(chunked_documents))]

    save_to_collection(M_embedd, meta, collection_name)
    return M_embedd,meta


def save_to_collection (M_embedd, meta, collection_name:str):
    collection = example.chroma_client.get_or_create_collection(collection_name)
    for i in range(0,len(M_embedd[:,0])):
        collection.add(
            ids=[str(i)],
            embeddings=[M_embedd[i,:].tolist()],
            metadatas=[meta[i]]
        )

def embeddings_from_collection(collection_name:str):
    #get all vectors from collection and save them in matrix
    collection = example.chroma_client.get_or_create_collection(collection_name)
    stored = collection.get(
        ids= [str(i) for i in range(0,collection.count())],
        include=["embeddings", "metadatas"]
    )
    embedds = stored["embeddings"]
    meta = stored["metadatas"]

    pairsort(meta,embedds)

    print("getting vectors: ", len(meta))

    M_embedd = np.zeros((len(embedds),EMBEDDING_DIM))
    
    for i in range(0,len(embedds)):
        M_embedd[i,:]= embedds[i]

    return M_embedd, meta

def project_embedding(axis, vector):
    return np.dot(axis, vector)


#histogramm functions
def split_by_source(values, meta):
    legend = []
    split_data = []
    i=0
    i_old = 0
    while i<len(values):
        s = meta[i]["source"]
        legend.append(s)

        while (i<len(values)) and (meta[i]["source"] == s):
            i+=1

        split_data.append(values[i_old:i])
        i_old = i
    return split_data, legend

def pairsort(meta, embedds):
    pairt=[(meta[i]["source"],embedds[i]) for i in range(0,len(embedds))]
    pairt.sort()

    for i in range(0,len(embedds)):
        meta[i]= pairt[i][0]
        embedds = pairt[i][1]

def show_hist(values):
    print("starting histogramm")
    # Plotting a basic histogram
    plt.hist(values, bins=30, color='skyblue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Basic Histogram')

    # Display the plot
    plt.show()

def show_stacked_hist(values, meta):
    
    split_data, legend = split_by_source(values, meta)

    plt.hist(split_data, bins=30, stacked=True, edgecolor='black')
 
    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Stacked Histogram')
    
    # Adding legend
    plt.legend(legend)
    
    # Display the plot
    plt.show()
    

def run_projection():
    ref1 = "./data/psych"
    ref2 = "./data/math"
    data = "./data/mixed-articles"
    
    
    #M_embedd, meta = new_load_embeddings(source=ref1, collection_name="ref1")
    M_embedd, meta = embeddings_from_collection("ref1")
    avg_psych = avarage_embedding(M_embedd)
    print("avg: ", avg_psych)
    
    
    #M_embedd, meta = new_load_embeddings(source=ref2, collection_name="ref2")
    M_embedd, meta = embeddings_from_collection("ref2")
    avg_math = avarage_embedding(M_embedd)
    print("avg: ", avg_math)

    axis = create_axis(avg_psych, avg_math)
    print("axis: ", axis)

    M_embedd, metadata = new_load_embeddings(source=data, collection_name="data-mixed-articles")
    #M_embedd, metadata = embeddings_from_collection("data-mixed-articles")
    #M_embedd, metadata = embeddings_from_collection("data-math-heavy-articles")
    
    values = np.zeros(len(M_embedd[:,0]))

    for i in range(0, len(M_embedd[:,0])):
        z = project_embedding(axis, M_embedd[i, :])
        values[i] = z  # saves all values

    show_stacked_hist(values, metadata)
  

def run_normalized_projection():
    ref1 = "./data/psych"
    ref2 = "./data/math"
    data = "./data/mixed-articles"
    
    
    #M_embedd, meta = new_load_embeddings_n(source=ref1, collection_name="ref1-n")
    M_embedd, meta = embeddings_from_collection("ref1-n")
    avg_psych = avarage_embedding_n(M_embedd)
    print("avg: ", avg_psych)
    
    
    #M_embedd, meta = new_load_embeddings_n(source=ref2, collection_name="ref2-n")
    M_embedd, meta= embeddings_from_collection("ref2-n")
    avg_math = avarage_embedding_n(M_embedd)
    print("avg: ", avg_math)

    axis = create_axis_n(avg_psych, avg_math)
    print("axis: ", axis)

    #M_embedd, metadata = new_load_embeddings_n(source=data, collection_name="data-mixed-articles-n")
    M_embedd, metadata = embeddings_from_collection("data-mixed-articles-n")
    #M_embedd, metadata = embeddings_from_collection("data-math-heavy-articles-n")
    
    # project all vectors and create histogram
    values = np.zeros(len(M_embedd[:,0]))

    print("starting projections")
    for i in range(0, len(M_embedd[:,0])):
        z = project_embedding(axis, M_embedd[i, :])
        values[i] = z  # saves all values

    z = project_embedding(axis,avg_psych)
    y = project_embedding(axis, avg_math)
    
    print("psych avg:", z)
    print("math avg:", y)

    show_stacked_hist(values, metadata)

if __name__ == '__main__':
   
    example.chroma_client.delete_collection("data-mixed-articles")

    run_projection()
     
   


   