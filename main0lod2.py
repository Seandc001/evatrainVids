#1.1
# import required functions, classes
from autollm import AutoQueryEngine
from autollm.utils.document_reading import read_files_as_documents
from dify_client import CompletionClient, ChatClient
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Set your environment variables
os.environ["OPENAI_API_KEY"] = "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"
os.environ["DIFY_API_KEY"] = "app-eOINKvOPFer53wEl6kw5Mqfj"  # Replace with your Dify API key

# Read documents for AutoQueryEngine
required_exts = [".pdf"]
documents = read_files_as_documents(input_dir="evaDocs", required_exts=required_exts)

# AutoQueryEngine setup
system_prompt = "You are a friendly ai assistant known as Eva, that help users find the most relevant and accurate answers to their questions based on the documents you have access to."
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
enable_cost_calculator = True

# llm params
model = "gpt-3.5-turbo"

# vector store params
vector_store_type = "LanceDBVectorStore"
uri = "./.lancedb"
table_name = "vectors"

# service context params
chunk_size = 1024

# query engine params
similarity_top_k = 5

llm_params = {"model": model}
vector_store_params = {"vector_store_type": vector_store_type, "uri": uri, "table_name": table_name}
service_context_params = {"chunk_size": chunk_size}
query_engine_params = {"similarity_top_k": similarity_top_k}

query_engine = AutoQueryEngine.from_parameters(documents=documents, system_prompt=system_prompt, query_wrapper_prompt=query_wrapper_prompt, enable_cost_calculator=enable_cost_calculator, llm_params=llm_params, vector_store_params=vector_store_params, service_context_params=service_context_params, query_engine_params=query_engine_params)

# Initialize the CompletionClient for Dify
#completion_client = CompletionClient(os.environ["DIFY_API_KEY"])

# Initialize the ChatClient
chat_client = ChatClient(os.environ["DIFY_API_KEY"])

class DifyClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://perceptron.evasworld.net/v1"  # Updated base URL


@app.route('/query', methods=['POST'])
def query():
    user_input = request.json.get('query')
    response = query_engine.query(user_input).response
    return jsonify({"response": response})

@app.route('/dify_chat', methods=['POST'])
def dify_chat():
    try:
        data = request.json
        chat_response = chat_client.create_chat_message(
            inputs=data.get("inputs", {}),
            query=data["query"],
            user=data["user"],
            response_mode=data.get("response_mode", "blocking"),
            conversation_id=data.get("conversation_id")
        )
        chat_response.raise_for_status()
        result = chat_response.json()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/dify_completion', methods=['POST'])
def dify_completion():
    try:
        data = request.json
        completion_response = completion_client.create_completion_message(
            inputs=data.get("inputs", {}),
            query=data["query"],
            response_mode=data.get("response_mode", "blocking"),
            user=data["user"]
        )
        completion_response.raise_for_status()
        result = completion_response.json()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)

