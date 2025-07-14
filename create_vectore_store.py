# Before running this cell, ensure that the pdf source files are first placed in a folder named "data" in the current working directory
import PyPDF2
import os
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import (
    Document,
    VectorStoreIndex,

)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
doc_names = "./data"
texts=[]
for items in os.listdir(doc_names):
    item_path = os.path.join(doc_names,items)
    text=""

    with open(item_path, 'rb') as file:
     reader = PyPDF2.PdfReader(file)
     for page in reader.pages:
        text += page.extract_text()
    texts.append(text)    

docs=[]
for index,text in enumerate(texts):
 d = Document(text=text,metadata = {"file": "h"+str(index), "name": "hr_pdf","_id":"hr_pdf"+str(index)})
 docs.append(d)
 
 # Define a text chunking procedure
text_chunker = SentenceSplitter(chunk_size=256, chunk_overlap=8)

# Split the documnets into nodes
nodes = text_chunker.get_nodes_from_documents(docs)

# Load a model for embedding the text
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")    

output_dir = './vector_store/'
len(nodes)

index = VectorStoreIndex(
        nodes,
        embed_model=embed_model,
        show_progress=True
    )

    ## Save embeddings with a storage context.
index.storage_context.persist(persist_dir = output_dir)
print(f"Index created and saved to {output_dir}")