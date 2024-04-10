import example
import numpy as np
import matplotlib.pyplot as plt
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

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


def data_embedd_n(chunked_documents):
    M_embedd = np.zeros((len(chunked_documents), EMBEDDING_DIM))

    for i in range(0, len(chunked_documents)):
        # getting embedding and adding vector to collection
        embedding = example.get_embedding(chunked_documents[i].page_content)

        embedding = embedding / np.linalg.norm(embedding)

        M_embedd[i, :] = embedding  # every row is one embedding vector

    return M_embedd


def average_embedding(M_embedd):
    avg = M_embedd.mean(0)  # mean row
    return avg


def average_embedding_n(M_embedd):
    avg = M_embedd.mean(0)  # mean row
    avg = avg / np.linalg.norm(avg)
    return avg


def create_axis(vec_1, vec_2):
    return vec_2 - vec_1


def create_axis_n(vec_1, vec_2):
    x = vec_2 - vec_1
    return x / np.linalg.norm(x)


def new_load_embeddings(source: str, collection_name: str):
    chunked_documents = split_pdf_document(source)
    print("chunked documents: ", len(chunked_documents))
    M_embedd = data_embedd(chunked_documents)

    meta = [chunked_documents[i].metadata for i in range(0, len(chunked_documents))]

    save_to_collection(M_embedd, meta, collection_name)
    return M_embedd, meta


def new_load_embeddings_n(source: str, collection_name: str):
    chunked_documents = split_pdf_document(source)
    print("chunked documents: ", len(chunked_documents))
    M_embedd = data_embedd_n(chunked_documents)

    meta = [chunked_documents[i].metadata for i in range(0, len(chunked_documents))]

    save_to_collection(M_embedd, meta, collection_name)
    return M_embedd, meta


def save_to_collection(M_embedd, meta, collection_name: str):
    collection = example.chroma_client.get_or_create_collection(collection_name)
    for i in range(0, len(M_embedd[:, 0])):
        collection.add(
            ids=[str(i)],
            embeddings=[M_embedd[i, :].tolist()],
            metadatas=[meta[i]]
        )


def embeddings_from_collection(collection_name: str):
    # get all vectors from collection and save them in matrix
    collection = example.chroma_client.get_or_create_collection(collection_name)
    stored = collection.get(
        ids=[str(i) for i in range(0, collection.count())],
        include=["embeddings", "metadatas"]
    )
    embedds = stored["embeddings"]
    meta = stored["metadatas"]

    print("getting vectors: ", len(meta))

    M_embedd = np.matrix(embedds)

    """
    M_embedd = np.zeros((len(embedds), EMBEDDING_DIM))
    for i in range(0, len(embedds)):
        M_embedd[i, :] = embedds[i]
    """
    return M_embedd, meta


def project_embedding(axis, vector):
    return np.dot(axis, vector)


# histogramm functions
def split_by_attribute(values, meta, attribute: str):
    pairsort(meta, values, attribute)
    legend = []
    split_data = []
    i = 0
    i_old = 0
    while i < len(values):
        s = meta[i]
        legend.append(s)

        while (i < len(values)) and (meta[i] == s):
            i += 1

        split_data.append(values[i_old:i])
        i_old = i
    return split_data, legend


def pairsort(meta, embedds, attribute: str):
    pairt = [(meta[i][attribute], embedds[i]) for i in range(0, len(meta))]
    pairt.sort()

    for i in range(0, len(embedds)):
        meta[i] = pairt[i][0]
        embedds[i] = pairt[i][1]


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


def get_average_attribute(num_bins: int, values, meta, attribute: str):
    # pairsort values, meta by values
    pairt = [(values[i], meta[i][attribute]) for i in range(0, len(values))]
    pairt.sort()

    for i in range(0, len(values)):
        values[i] = pairt[i][0]
        meta[i] = pairt[i][1]

    #print(values)
    #calculate avarage for each bin
    width = (values[len(values) - 1] - values[0]) / num_bins
    print("width: ", width)
    stop = values[0] + width
    avgs = []
    i = 0  # iterator
    c = 0  # number of entries in bin
    sum = 0  # sum over current bin
    
    while i < len(values):
        #print(stop, values[i])
        while values[i] < stop or i == len(values) - 1:
            sum += meta[i]
            c += 1
            i += 1
            if i == len(values):
                break

        avgs.append(sum / c) if c != 0 else avgs.append(0)
        c = 0
        sum = 0
        stop += width

    return avgs
    # adjust last sum


def show_stacked_hist(values, meta, attribute, num_bins):
    #avgs = get_average_attribute(num_bins, values, meta.copy(), attribute="wls")
    print("avgs: ", avgs)

    split_data, legend = split_by_attribute(values, meta, attribute)
    #print(legend)
    #print(split_data)
    plt.hist(split_data, bins=num_bins, stacked=True, edgecolor='black')

    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Stacked Histogram')

    # Adding legend
    plt.legend(legend)

    # try showing average wls for every bin
    """
    try:
        w= 0.5/num_bins
        bin_centers = [(i*w-0.25) for i in range(0,num_bins)]
        for avg, bin_center in zip(avgs, bin_centers):
            plt.annotate(text=str(int(avg)), xy=(bin_center, 0.5))
    except Exception as error:
        print(error)
    """
    # Display the plot
    plt.show()


# running th projections
def run_projection():
    ref1 = "./data/psych"
    ref2 = "./data/math"
    data = "./data/mixed-articles"

    # M_embedd, meta = new_load_embeddings(source=ref1, collection_name="ref1")
    M_embedd, meta = embeddings_from_collection("ref1", )
    avg_psych = average_embedding(M_embedd)
    print("avg: ", avg_psych)

    # M_embedd, meta = new_load_embeddings(source=ref2, collection_name="ref2")
    M_embedd, meta = embeddings_from_collection("ref2")
    avg_math = average_embedding(M_embedd)
    print("avg: ", avg_math)

    axis = create_axis(avg_psych, avg_math)
    print("axis: ", axis)

    # M_embedd, metadata = new_load_embeddings(source=data, collection_name="data-mixed-articles")
    M_embedd, metadata = embeddings_from_collection("data-mixed-articles")
    # M_embedd, metadata = embeddings_from_collection("data-math-heavy-articles")

    values = np.zeros(len(M_embedd[:, 0]))

    for i in range(0, len(M_embedd[:, 0])):
        z = project_embedding(axis, M_embedd[i, :])
        values[i] = z  # saves all values

    z = project_embedding(axis, avg_psych)
    y = project_embedding(axis, avg_math)

    print("psych avg:", z)
    print("math avg:", y)

    show_stacked_hist(values, metadata, "source", num_bins=30)


def run_normalized_projection():
    ref1 = "./data/psych"
    ref2 = "./data/math"
    data = "./data/mixed-articles"

    # M_embedd, meta = new_load_embeddings_n(source=ref1, collection_name="ref1-n")
    M_embedd, meta = embeddings_from_collection("ref1-n")
    avg_psych = average_embedding_n(M_embedd)
    print("avg: ", avg_psych)

    # M_embedd, meta = new_load_embeddings_n(source=ref2, collection_name="ref2-n")
    M_embedd, meta = embeddings_from_collection("ref2-n")
    avg_math = average_embedding_n(M_embedd)
    print("avg: ", avg_math)

    axis = create_axis_n(avg_psych, avg_math)
    print("axis: ", axis)

    # M_embedd, metadata = new_load_embeddings_n(source=data, collection_name="data-mixed-articles-n")
    # M_embedd, metadata = embeddings_from_collection("data-mixed-articles-n")
    M_embedd, metadata = embeddings_from_collection("data-math-heavy-articles-n")

    # project all vectors and create histogram
    values = np.zeros(len(M_embedd[:, 0]))

    print("starting projections")
    for i in range(0, len(M_embedd[:, 0])):
        z = project_embedding(axis, M_embedd[i, :])
        values[i] = z  # saves all values

    z = project_embedding(axis, avg_psych)
    y = project_embedding(axis, avg_math)

    print("psych avg:", z)
    print("math avg:", y)

    show_stacked_hist(values, metadata, "source", num_bins=30)


if __name__ == '__main__':
    run_normalized_projection()
