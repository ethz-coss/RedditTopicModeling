from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

#returns a list of documents with .page_content being the string and some metadata
def split_pdf_document(data_path: str):
    # load pdf documents under DATA_DIR path
    text_loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    loaded_documents = text_loader.load()
    print("documents loaded")
    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(loaded_documents)
    return chunked_documents

