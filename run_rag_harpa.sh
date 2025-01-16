echo "Starting Flask application..."

PERSIST_DIRECTORY="chroma_storage"
COLLECTION_NAME="documents_collection"

nohup python app.py --persist_directory "$PERSIST_DIRECTORY" --collection_name "$COLLECTION_NAME" > flask.log 2>&1 &

echo "Flask is running in the background."
echo "Logs are being saved in flask.log"