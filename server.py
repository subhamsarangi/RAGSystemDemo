from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

folder_path = "products/"

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
llm_model_name = "deepseek-r1:1.5b"

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

embedder = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

text_splitter = SemanticChunker(embedder)
documents = text_splitter.split_documents(docs)
print(len(documents), "documents")

# Create FAISS index and retriever
vector = FAISS.from_documents(documents, embedder)
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Set up the system and prompt templates
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

# Create the documents chain and retrieval chain
documents_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, documents_chain)

print(rag_chain, "rag_chain")


@app.get("/", response_class=HTMLResponse)
async def get_form():
    html_content = """
    <html>
        <head>
            <title>Ask a Question</title>
        </head>
        <body>
            <h1>Ask a Question</h1>
            <form action="/query" method="post">
                <label for="query">Enter your question:</label>
                <input type="text" id="query" name="query" required>
                <button type="submit">Submit</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/query")
async def answer_query(query: QueryRequest):
    try:
        response = rag_chain.invoke({"input": query.question})["answer"]
        return {"answer": response.split("</think>")[1].strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")

@app.post("/query", response_class=HTMLResponse)
async def answer_query(query: str = Form(...)):
    try:
        response = rag_chain.invoke({"input": query})["answer"]
        clean_response = response.split("</think>")[1].strip() if "</think>" in response else response.strip()
        return f"<h3>Question: {query}</h2><h3>Answer: {clean_response}</h2>"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")