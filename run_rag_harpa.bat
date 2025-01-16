@echo off
echo Starting Flask application...

set PERSIST_DIRECTORY=chroma_storage
set COLLECTION_NAME=documents_collection

start /B python app.py --persist_directory %PERSIST_DIRECTORY% --collection_name %COLLECTION_NAME% > flask.log 2>&1

echo Flask is running in the background.
ech
