from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
from llama_index.llms.groq import Groq
import os
from llama_index.core import Settings
from dotenv import load_dotenv
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
    get_response_synthesizer
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode

app = FastAPI(title="HR Assistant RAG API", 
              description="API for HR assistant using Retrieval Augmented Generation")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# Load environment variables and set up the models
load_dotenv()
key = os.getenv("GROQ_KEY")
if not key:
    raise ValueError("GROQ_KEY not found in environment variables")

# Initialize Groq LLM and embedding model
llm = Groq(model="llama3-8b-8192", api_key=key)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Set up the global settings
Settings.llm = llm

# Load the index from storage
output_dir = './vector_store/'
print(f"Loading existing index from {output_dir}...")
storage_context = StorageContext.from_defaults(persist_dir=output_dir)
index = load_index_from_storage(
    storage_context=storage_context,
    embed_model=embed_model
)
print("Index loaded successfully")

# Set up the query engine
TopK = 5

# Define the prompt template
qa_prompt_tmpl = PromptTemplate(
    "You are an expert assistant for an organization that helps the users to understand the ways this organization deals with things. You are asked a question by a employee for a certain task and you will be explaining it the regulations of the company for the scenario. You are expected to provide a detailed and accurate answer if the context is provided. If no context is provided, you must respond with 'Context not provided, unable to answer.'\n"
    "Always answer the query using the provided context information, and not prior knowledge.\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

# Define the retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=TopK,
)

# Set the form of context consolidation
response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.SIMPLE_SUMMARIZE)

# Build the improved query engine and set the template
query_engine = RetrieverQueryEngine.from_args(
    retriever,
    response_synthesizer=response_synthesizer,
)
query_engine.update_prompts(
     {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)

def get_rag_answer(query):
    """
    Process a query using the RAG system
    """
    response = query_engine.query(query)
    return str(response)

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query and return the RAG response
    """
    try:
        result = get_rag_answer(request.query)
        return QueryResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {"message": "HR Assistant RAG API is running. Use the /query endpoint to interact with the RAG system."}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9000, reload=True)