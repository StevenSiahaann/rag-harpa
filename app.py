from flask import Flask, request, jsonify
from flask_cors import CORS 
import warnings
import os
import sys
from collections import defaultdict
from sentence_transformers import SentenceTransformer,util
import google.generativeai as genai
from typing import List, Dict
import sys
import chromadb
from chromadb.utils import embedding_functions
from util.prepare_document import *
from util.prompt_design import *
from util.intent_example import *
from util.load_initial_rag_document import *
from util.check_session import *
from util.load_endpoint import *
import argparse
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()
conversation_state = defaultdict(dict)
chat_history = defaultdict(list)


warnings.filterwarnings("ignore", message="You are using the default legacy behaviour")
warnings.filterwarnings("ignore", message="It will be set to `False` by default.")
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")
persist_directory=os.getenv("PERSIST_DIRECTORY")

GEMINI_API_KEY=os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")
UPLOAD_FOLDER = os.path.join(os.getcwd(), '/app/uploads')
if not os.path.exists(UPLOAD_FOLDER):
    print("Creating folder...")
    os.makedirs(UPLOAD_FOLDER)
else:
    print("Folder already exists.")

app = Flask(__name__)
CORS(app)
collection = None
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


session_history: Dict[str, List[Dict[str, str]]] = {}
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
intent_embeddings = {key: embedding_model.encode(sentences) for key, sentences in intent_examples.items()}

model = genai.GenerativeModel("gemini-1.5-pro")
def validate_document_format(filename):
    valid_extensions = ('.pdf', '.ppt', '.docx', '.jpg', '.jpeg', '.png', '.txt', '.json', '.csv', '.pptx')
    return filename.endswith(valid_extensions)
def get_gemini_response(query: str, context: List[str],intent, session_id: str, extract_data: bool = False,documentID: str = None) -> str:
    history = session_history.get(session_id, [])
    def rerank_results(context: List[str], query: str) -> List[str]:
        scores = embedding_model.encode([query], convert_to_tensor=True)
        doc_scores = embedding_model.encode(context, convert_to_tensor=True)
        similarities = util.cos_sim(scores, doc_scores)        
        ranked_pairs = sorted(zip(context, similarities), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_pairs]
    if(intent=="cuti"):
        used_context=context
        prompt = build_combined_prompt(query, used_context, history)
    else:
        ranked_context = rerank_results(context, query)
        used_context = ranked_context[:3]
        prompt = build_combined_prompt(query, used_context, history)
    response = model.generate_content(prompt)
    response_text = response.text.strip().lower()
    session_history.setdefault(session_id, []).append({"query": query, "response": response_text,"context": used_context
})
    return response.text
@app.route('/v1/knowledge', methods=['POST'])
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

            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], doc_name)
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
            # pdf_path = os.path.join("uploads", doc_name)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], doc_name)

            all_pages = "\n".join([doc.page_content for doc in documents])

            pdfkit.from_string(all_pages, pdf_path)

            upload_path = pdf_path
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



    external_context = []
    try:
        detected_intent = detect_intent(embedding_model,intent_embeddings,query)

        if detected_intent == "cuti":
            url= 'https://hrp02-dev-be-v3.harpa-go.com:8080/apps/accrualPlans/getNetEntitleMobile/?people_uuid=ee140c35-6f32-426d-9c51-7b74d05160aa&effective_date=2025-01-16'
            headers={
            "Authorization": f"JWT {bearer_token}",
            "Access-Function": request.headers.get('Access-Function'),
            "Access-Org": request.headers.get('Access-Org'),
            "Access-Role": request.headers.get('Access-Role')
            }
            response=load_endpoint(url,headers)
            if "error" in response:
                external_context.append(f"Data gaji kamu belum tersedia, coba kontak tim HARPA untuk info lebih lanjut.")
            elif response and isinstance(response, dict):
                external_context.append(f"Data gaji terbaru: {response["net_entitlement"]} {response["satuan"]}")

        context_results = collection.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "metadatas"],
            where={"document_id": document_id} if document_id else None  
        )

        if not context_results["documents"]:
            return jsonify({"error": "No relevant information found in the database."}), 404
        context = [doc for docs in context_results["documents"] for doc in docs] + external_context
        metadata = [meta for metas in context_results["metadatas"] for meta in metas]

        if not all(isinstance(c, str) for c in context):
            return jsonify({"error": "Context is not in the expected format."}), 500

        response = get_gemini_response(query, context,detected_intent, session_id, documentID=document_id)
        chat_history[user_id].append({"query": query, "response": response})


        references = [{
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


def main(collection_name: str = "documents_collection") -> None:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load documents into a Chroma collection")
    # parser.add_argument("--persist_directory", type=str, default="chroma_storage", help="Directory to store the Chroma collection")
    parser.add_argument("--collection_name", type=str, default="documents_collection", help="Name of the Chroma collection")
    args = parser.parse_args()
    main(collection_name=args.collection_name)
