from flask import Flask, request, jsonify
from flask_cors import CORS 
from transformers import T5ForConditionalGeneration, T5Tokenizer
import warnings
import os
import sys
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from util.prepare_document import *
from util.prompt_design import *
from util.load_initial_rag_document import *
from util.check_session import *
import argparse
from urllib.parse import urlparse
import torch

from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()
# base_url = os.getenv("BASE_URL")
conversation_state = defaultdict(dict)
chat_history = defaultdict(list)


warnings.filterwarnings("ignore", message="You are using the default legacy behaviour")
warnings.filterwarnings("ignore", message="It will be set to `False` by default.")
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")

GEMINI_API_KEY=os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    print("Creating folder...")
    os.makedirs(UPLOAD_FOLDER)
else:
    print("Folder already exists.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)
collection = None
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
# flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

session_history: Dict[str, List[Dict[str, str]]] = {}
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = genai.GenerativeModel("gemini-1.5-pro")
# headers = {
#     "x-team": "steven"
# }
def validate_document_format(filename):
    valid_extensions = ('.pdf', '.ppt', '.docx', '.jpg', '.jpeg', '.png', '.txt', '.json', '.csv', '.pptx')
    return filename.endswith(valid_extensions)
def get_gemini_response(query: str, context: List[str], session_id: str, extract_data: bool = False,documentID: str = None) -> str:
    def rerank_results(context: List[str], query: str) -> List[str]:
        scores = embedding_model.encode([query], convert_to_tensor=True)
        doc_scores = embedding_model.encode(context, convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(scores, doc_scores)        
        ranked_pairs = sorted(zip(context, similarities), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_pairs]
    history = session_history.get(session_id, [])
    ranked_context = rerank_results(context, query)
    truncated_context = ranked_context[:3]
    prompt = build_combined_prompt(query, truncated_context, history)
    response = model.generate_content(prompt)
    response_text = response.text.strip().lower()
    session_history.setdefault(session_id, []).append({"query": query, "response": response_text,"context": truncated_context
})
    return False, response.text
@app.route('/v1/knowledge/', methods=['POST'])
def upload_document():
    try:
        authorization_header = request.headers.get('Authorization')

        if not authorization_header:
            return jsonify({"error": "Missing Authorization header"}), 401

        if not authorization_header.startswith("Bearer "):
            return jsonify({"error": "Invalid Authorization header format"}), 401
        bearer_token = authorization_header.split(" ")[1]
        result = decode_and_check_exp(bearer_token)

        if "error" in result:
            return jsonify({"error": f"{result["error"]}"}), 401

        text = None
        doc_name = None
        if request.args.get('type')=='file':
            if 'file' not in request.files:
                return jsonify({"error": "No document uploaded."}), 400

            document = request.files['file']
            doc_name = document.filename
            if not validate_document_format(doc_name):
                return jsonify({"error": "Invalid document format. Only PDF, PPT, DOCX, and images are supported."}), 400

            upload_path = os.path.join("uploads", doc_name)
            text = extract_text_from_file(upload_path)
            document.save(upload_path)

        elif(request.args.get('type')=='url'):
            if not request.json.get('url'):
                return jsonify({"error": "No URL provided."}), 400
            url = request.json.get('url')
            documents = get_page([url])

            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace(".", "_")
            path = parsed_url.path.replace("/", "_").replace(":", "_")

            safe_url = f"{domain}{path}"
            doc_name = f"url_content_{safe_url}_{os.urandom(6).hex()}.pdf"
            pdf_path = os.path.join("uploads", doc_name)

            all_pages = "\n".join([doc.page_content for doc in documents])

            pdfkit.from_string(all_pages, pdf_path)

            upload_path = pdf_path
            text = extract_text_from_file(upload_path)
        elif request.args.get('type') == 'curl':
            curl_command = request.json.get('curl')
            if not curl_command:
                return jsonify({"error": "No curl command provided."}), 400
            url = extract_url_from_curl(curl_command)
            if not url:
                return jsonify({"error": "Invalid curl command or URL missing."}), 400
            documents = get_page([url])
            doc_name = f"curl_content_{os.urandom(6).hex()}.txt"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], doc_name)
            with open(upload_path, 'w', encoding='utf-8') as f:
                for doc in documents:
                    f.write(doc.page_content + '\n')
            text = extract_text_from_file(upload_path)


        if not text:
            return jsonify({"error": "Failed to extract text from the document."}), 400
        document_id = "doc_" + os.urandom(6).hex()
        lines = text.splitlines()
        documents = [line for line in lines if line.strip()]
        metadatas = [{"document_id": document_id, "filename": doc_name, "line_number": i} for i, _ in enumerate(documents, 1)]
        collection.add(
            ids=[f"{document_id}_{i}" for i in range(len(documents))],
            documents=documents,
            metadatas=metadatas
        )
        return jsonify({"message": "Document uploaded and processed successfully.", "document_id": document_id}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500





@app.route('/v1/chat', methods=['POST'])
def chat():
    authorization_header = request.headers.get('Authorization')

    if not authorization_header:
        return jsonify({"error": "Missing Authorization header"}), 401

    if not authorization_header.startswith("Bearer "):
        return jsonify({"error": "Invalid Authorization header format"}), 401
    bearer_token = authorization_header.split(" ")[1]
    result = decode_and_check_exp(bearer_token)

    if "error" in result:
        return jsonify({"error": f"{result["error"]}"}), 401
    user_id = result.get("user_id")
    if not user_id:
        return jsonify({"error": "Invalid token, user_id missing"}), 401

    data = request.json
    session_id = request.headers.get('x-session-id')
    query = data.get('query')
    document_id = data.get('document_id')
    if not query or not isinstance(query, str):
        return jsonify({"error": "Query is required and must be a string oooh."}), 400
    if query is None:
        return jsonify({"error": "Query is missing or None."}), 400
    if not query or not isinstance(query, str):
        return jsonify({"error": "Query is required and must be a string."}), 400
    try:
        context_results = collection.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "metadatas"],
            where={"document_id": document_id} if document_id else None  
        )

        if not context_results["documents"]:
            return jsonify({"error": "No relevant information found in the database."}), 404
        context = [doc for docs in context_results["documents"] for doc in docs]
        metadata = [meta for metas in context_results["metadatas"] for meta in metas]

        if not all(isinstance(c, str) for c in context):
            return jsonify({"error": "Context is not in the expected format."}), 500

        action_flag, response = get_gemini_response(query, context, session_id, documentID=document_id)
        chat_history[user_id].append({"query": query, "response": response})


        references = [{
            "excerpt": doc[:200],  # Show the first 200 characters of the relevant context
            "line_number": meta["line_number"],
            "document_name": meta["filename"]
        } for doc, meta in zip(context, metadata)]

        response_data = {
            "harpa_rag_message": response,
            "references": references
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

@app.route('/v1/chat/history', methods=['GET'])
def get_chat_history():
    authorization_header = request.headers.get('Authorization')

    if not authorization_header:
        return jsonify({"error": "Missing Authorization header"}), 401

    if not authorization_header.startswith("Bearer "):
        return jsonify({"error": "Invalid Authorization header format"}), 401

    bearer_token = authorization_header.split(" ")[1]
    result = decode_and_check_exp(bearer_token)

    if "error" in result:
        return jsonify({"error": f"{result['error']}"}), 401

    user_id = result.get("user_id")
    if not user_id:
        return jsonify({"error": "Invalid token, user_id missing"}), 401

    user_history = chat_history.get(user_id, [])

    return jsonify({"user_id": user_id, "chat_history": user_history}), 200

@app.route('/v1/knowledge/documents', methods=['GET'])
def list_documents():
    authorization_header = request.headers.get('Authorization')

    if not authorization_header:
        return jsonify({"error": "Missing Authorization header"}), 401

    if not authorization_header.startswith("Bearer "):
        return jsonify({"error": "Invalid Authorization header format"}), 401

    bearer_token = authorization_header.split(" ")[1]
    result = decode_and_check_exp(bearer_token)

    if "error" in result:
        return jsonify({"error": f"{result['error']}"}), 401

    try:
        all_documents = collection.get(include=["metadatas"])
        
        if not all_documents["metadatas"]:
            return jsonify({"message": "No documents found in the collection."}), 200
        unique_documents = {}        
        for metadata in all_documents["metadatas"]:
            doc_id = metadata.get("document_id", "Unknown ID")
            if doc_id not in unique_documents:
                unique_documents[doc_id] = {
                    "document_id": doc_id,
                    "filename": metadata.get("filename", "Unknown Filename"),
                    "line_count": 0
                }
            unique_documents[doc_id]["line_count"] += 1

        document_list = list(unique_documents.values())
        return jsonify({"documents": document_list}), 200


    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


def main(collection_name: str = "documents_collection", persist_directory: str = ".") -> None:
    global collection

    client = chromadb.PersistentClient(path=persist_directory)
    embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_key=HUGGING_FACE_KEY
    )

    try:
        collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
        print(f"Loaded existing collection: {collection_name}")
    except Exception as e:
        print(f"Collection '{collection_name}' not found. Creating a new collection.")
        collection = client.create_collection(name=collection_name, embedding_function=embedding_function)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))

    # app.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load documents into a Chroma collection")
    parser.add_argument("--persist_directory", type=str, default="chroma_storage", help="Directory to store the Chroma collection")
    parser.add_argument("--collection_name", type=str, default="documents_collection", help="Name of the Chroma collection")
    args = parser.parse_args()
    main(collection_name=args.collection_name, persist_directory=args.persist_directory)
