## RAG (Retrieval Augmented Generation)
Way to add data to an llm model without retraining it. needed when llm model has a cutoff date or for using sensitive data.

## Vector Embeddings
A format of data which holds semantic information.

## Semantic Search
Way to identiify things which are similar in meaning, not in the characters. eg: dog ~= cat; dog !~= dot.

## Useful commands to run
```
ollama --version
ollama list
ollama pull deepseek-r1:1.5b
ollama pull llama3.2:1bV

poetry init --no-interaction
poetry add faiss-cpu langchain langchain-community langchain_experimental sentence-transformers langchain-ollama langchain-huggingface fastapi uvicorn python-multipart
```
`python script.py` 

or

`uvicorn server:app --reload`

