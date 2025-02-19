"""RAG_with_DeepSeek_R1

Original file is located at
    https://colab.research.google.com/drive/19TzsHlmONaFXmuFC6DX6aeSmZE2YNASw
"""


import os
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain



folder_path = "products/"

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
llm_model_name = "deepseek-r1:1.5b"
# llm_model_name = "llama3.2:1b"

def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
    return documents

docs = load_documents_from_folder(folder_path)

print(len(docs), "docs")
embedder = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

print(embedder, "embedder")

text_splitter = SemanticChunker(embedder)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
print(len(documents), "documents")

# Create FAISS index
vector = FAISS.from_documents(documents, embedder)
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())
# retriever = vectorstore.as_retriever()


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


llm = OllamaLLM(model=llm_model_name)

documents_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, documents_chain)

def answer_query(question):
    response = rag_chain.invoke({"input": question})["answer"]
    return response.split("</think>")[1].strip()

user_question = "What is the main technology behind gravitas sneaker?"
answer = answer_query(user_question)
print("Answer:", answer)

# !rm -rf /usr/local/bin/ollama