from langchain.llms.base import LLM
from huggingface_hub import InferenceClient
from typing import Optional, List
import os


class HuggingFaceLLM(LLM):
    client: InferenceClient
    model: str = "tiiuae/falcon-7b-instruct"
    temperature: float = 0.7
    max_new_tokens: int = 512

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.text_generation(
            prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            stop_sequences=stop,
        )
        return response

    @property
    def _llm_type(self) -> str:
        return "huggingface_custom_llm"


def setup_qa_chain(docs):
    from langchain.chains import RetrievalQA
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.docstore.document import Document
    from langchain.vectorstores.faiss import FAISS

    # Use your Hugging Face API token
    client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    llm = HuggingFaceLLM(client=client)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI assistant. Use the following context to answer the question.

        Context: {context}

        Question: {question}

        Answer:"""
    )

    # Load embedding model and create vector DB
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    vectordb = FAISS.from_documents(split_docs, embeddings)

    retriever = vectordb.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt})
    return chain


def get_answer(chain, query):
    return chain.run(query)
