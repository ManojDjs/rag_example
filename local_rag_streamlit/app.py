import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama

# Setup FAISS + Ollama
@st.cache_resource
def load_qa_chain():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 👇 TRUST YOUR OWN INDEX IF YOU CREATED IT
    db = FAISS.load_local(
        folder_path="faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # <-- this is required now
    )
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="mistral")  # or any other local model you've pulled
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

st.title("📄 Local Document Q&A")
st.markdown("Ask questions based on your uploaded documents (FAISS + Ollama)")

qa_chain = load_qa_chain()

query = st.text_input("💬 Enter your question:")
if query:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(query)
        st.success("🧠 Answer:")
        st.write(answer)
