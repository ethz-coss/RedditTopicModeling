from chromadb import EmbeddingFunction, Documents
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings
import numpy as np
import example



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

    query_text = "Harry Potter"
    embedding_vector = example.get_embedding(query_text)
    print(embedding_vector)

    print("collection count: ", db._collection.count())
    retriever1 = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.99})

    docs = retriever1.get_relevant_documents(query_text)
    print(len(docs))
    # print(docs[0].page_content)

    retriever2 = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever2.get_relevant_documents(query_text)
    print(len(docs))
    print(docs)

    docs = db.asimilarity_search_by_vector(
        embedding= embedding_vector.tolist(),
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.99})

    print(docs)


