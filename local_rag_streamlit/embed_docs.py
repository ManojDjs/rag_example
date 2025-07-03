import os
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

DOCS_DIR = "documents"
INDEX_DIR = "faiss_index"

def load_documents():
    docs = []
    for filename in os.listdir(DOCS_DIR):
        print(f"📄 Loading document: {filename}")
        path = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif filename.endswith(".txt"):
            docs.extend(TextLoader(path).load())
        elif filename.endswith(".docx"):
            docs.extend(Docx2txtLoader(path).load())
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def embed_and_store(docs):
    print("🔄 Generating embeddings and saving FAISS index...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    
    if os.path.exists(index_path):
        db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        print("✅ Loaded existing FAISS index.")
    else:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(INDEX_DIR)
        print("✅ Created and saved new FAISS index.")
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    print("✅ Embeddings stored at:", INDEX_DIR)

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    embed_and_store(chunks)
