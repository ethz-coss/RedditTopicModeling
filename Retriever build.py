import numpy as np
from chromadb import EmbeddingFunction, Documents
import example
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings

import example
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb


class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            vector = example.get_embedding(doc)  # should be passsing a string not a doc
            embeddings.append(vector.tolist())
        # embed the documents somehow
        return embeddings

    def embed_documents(self, texts):
        return self.__call__(texts)

    def embed_query(self, query):
        em = self.__call__([query])[0]
        print(em)
        return em


if __name__ == '__main__':
    db = Chroma(
        client=example.chroma_client,
        collection_name="harry_potter_100",
        embedding_function=MyEmbeddingFunction()
    )

    query_text = "Philosopher’s Stone"
    print(example.get_embedding(query_text))

    retriever1 = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.99})

    docs = retriever1.get_relevant_documents(query_text)
    print(len(docs))
    #print(docs[0].page_content)


    retriever2 = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever2.get_relevant_documents(query_text)
    print(len(docs))
    print(docs)
