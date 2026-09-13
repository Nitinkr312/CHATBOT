import streamlit as st
from document_loader import load_and_split_docs
from qa_engine import setup_qa_chain, get_answer
import os
from langchain.llms import HuggingFaceHub

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
)


st.title("📄 Internal Docs Q&A Chatbot")
uploaded_file = st.file_uploader("Upload your internal document", type=["pdf", "txt", "md"])

if uploaded_file:
    file_name = uploaded_file.name
    file_path = os.path.join("temp", file_name)

    os.makedirs("temp", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Document uploaded and ready for questions!")
    docs = load_and_split_docs(file_path)
    qa = setup_qa_chain(docs)

    query = st.text_input("Ask a question about the document:")

    if query:
        answer = get_answer(qa, query)
        st.markdown("**Answer:**")
        st.write(answer)
