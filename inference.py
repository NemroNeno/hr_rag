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





load_dotenv()
key=os.getenv("GROQ_KEY")

llm = Groq(model="llama3-8b-8192", api_key=key)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def gpt_and_rag_answers(query,query_engine):
    rag_response = query_engine.query(query)
    return rag_response

output_dir = './vector_store/'
print(f"Loading existing index from {output_dir}...")
storage_context = StorageContext.from_defaults(persist_dir = output_dir)
index = load_index_from_storage(
    storage_context=storage_context,
    embed_model=embed_model
)
print("Index loaded successfully")

Settings.llm = llm



TopK = 5

## Improve the prompting template to give more verbose answers
qa_prompt_tmpl = PromptTemplate(
    "You are an expert assistant for an organization that helps the users to understand the ways this organization deals with things. You are asked a question by a employee for a certain task and you will be explaining it the regulations of the company for the scenario.  You are expected to provide a detailed and accurate answer if the context is provided. If no context is provided, you must respond with 'Context not provided, unable to answer.'\n"
    "Always answer the query using the provided context information, and not prior knowledge.\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

###################

## Define the larger-k retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=TopK,
)

## Set the form of context consolidation
response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.SIMPLE_SUMMARIZE)

## Build the improved query engine and set the template to the new one.
custom_query_engine = RetrieverQueryEngine.from_args(
    retriever,
    response_synthesizer=response_synthesizer,
)
custom_query_engine.update_prompts(
     {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)


query = "What are the company holidays provided?"
rag = gpt_and_rag_answers(query, custom_query_engine)
print(f"\n\n############\nRAG response:\n{rag}")


