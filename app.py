# import required functions, classes
import re

from autollm import AutoQueryEngine
from autollm.utils.document_reading import read_github_repo_as_documents, read_files_as_documents 
#, read_web_as_documents
import os
import gradio as gr
import requests
import responses
from flask import Flask, request, jsonify
from flask_cors import CORS

import threading

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


os.environ["OPENAI_API_KEY"] = "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"

#Git method
#git_repo_url = "https://github.com/ultralytics/ultralytics.git"
#relative_folder_path = "docs"   # relative path from the repo root to the folder containing documents
#required_exts = [".md"]    # optional, only read files with these extensions

#documents = read_github_repo_as_documents(git_repo_url=git_repo_url, relative_folder_path=relative_folder_path, required_exts=required_exts)

required_exts = [".pdf"]
documents = read_files_as_documents(input_dir="evaDocs", required_exts=required_exts)

# llm params
llm_model = "gpt-3.5-turbo"
llm_max_tokens = 512
llm_temperature = 0.1

# service_context_params
system_prompt = """
You are an friendly ai assistant that help users find the most relevant and accurate answers
to their questions based on the documents you have access to.
When answering the questions, mostly rely on the info in documents.
"""
query_wrapper_prompt = '''
The document information is below.
---------------------
{context_str}
---------------------
Using the document information and mostly relying on it,
answer the query.
Query: {query_str}
Answer:
'''
enable_cost_calculator = True,
embed_model = "default"  # ["default", "local"]
chunk_size = 512
context_window = 4096

# vector store params
vector_store_type = "LanceDBVectorStore"
lancedb_uri = "./.lancedb"
lancedb_table_name = "vectors"
enable_metadata_extraction = False

# query engine params
similarity_top_k = 3

query_engine = AutoQueryEngine.from_defaults(
    documents=documents,
    llm_model=llm_model,
    llm_max_tokens=llm_max_tokens,
    llm_temperature=llm_temperature,
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    enable_cost_calculator=enable_cost_calculator,
    embed_model=embed_model,
    chunk_size=chunk_size,
    context_window=context_window,
    vector_store_type=vector_store_type,
    lancedb_uri=lancedb_uri,
    lancedb_table_name=lancedb_table_name,
    enable_metadata_extraction=enable_metadata_extraction,
    similarity_top_k=similarity_top_k
)

#old
#query_engine = AutoQueryEngine.from_defaults(documents=documents)

def greet(query):
    return query_engine.query(query).response

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    

@app.route('/query', methods=['POST'])
def query():
    user_input = request.json.get('query')
    response = query_engine.query(user_input).response
    return jsonify({"response": response})



#@app.route('/query_advanced', methods=['POST'])
#def query():
#    user_input = request.json.get('query')
#    response = query_engine.query(user_input).response

    # Define the regex pattern
#    pattern = r"\*\*\*.*?@@@@"

    # Search for the pattern in the response
#    match = re.search(pattern, response)
#    link = None
#    if match:
        # Extract the link
#        link = match.group(0)
        # Remove the link from the response
#        response = re.sub(pattern, "", response)

    # Return the modified response and the extracted link (if any)
#    return jsonify({"response": response, "link": link if link else "No link found"})

# Thread to run Flask app
#def run_flask():
#    app.run(host='0.0.0.0', port=5001)
    
#def run_gradio():
#    demo.launch(server_port=7860, server_name='0.0.0.0')  # Make sure to specify server_name

# Running both Flask and Gradio in separate threads
#threading.Thread(target=run_flask).start()
#threading.Thread(target=run_gradio).start()

# Thread to run Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
