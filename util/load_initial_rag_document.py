from util.prepare_document import *
import pdfkit
def load_initial_knowledge(collection):
    """Load initial knowledge into the collection."""
    try:
        initial_data = [
            {"type": "file", "path": "./uploads/20241220 Peraturan-Perusahaan.pdf"},
            {"type": "url", "url": "https://www.harpa-go.com/"},
            {"type": "url", "url": "https://www.harpa-go.com/blog/categories/event"},
            {"type": "url", "url": "https://www.harpa-go.com/our-service"},
            {"type": "url", "url": "https://www.harpa-go.com/core-hr"},
            {"type": "url", "url": "https://www.harpa-go.com/time-and-attendance"},
            {"type": "url", "url": "https://www.harpa-go.com/benefit"},
            {"type": "url", "url": "https://www.harpa-go.com/payroll"},
            {"type": "url", "url": "https://www.harpa-go.com/analytics"},
            {"type": "url", "url": "https://www.harpa-go.com/self-service-hr"}
        ]
        for item in initial_data:
            if item["type"] == "file":
                file_path = item["path"]
                text = extract_text_from_file(file_path)
                doc_name = os.path.basename(file_path)
            elif item["type"] == "url":
                documents = get_page([item["url"]])
                text = " ".join([doc.page_content for doc in documents])
                doc_name = f"url_content_{os.urandom(6).hex()}.pdf"
                pdf_path = os.path.join("uploads", doc_name)
                pdfkit.from_string(text, pdf_path)
                text = extract_text_from_file(pdf_path)

            if text:
                document_id = "doc_" + os.urandom(6).hex()
                lines = text.splitlines()
                document_chunks = [line for line in lines if line.strip()]
                metadatas = [
                    {"document_id": document_id, "filename": doc_name, "line_number": i}
                    for i, _ in enumerate(document_chunks, 1)
                ]
                collection.add(
                    ids=[f"{document_id}_{i}" for i in range(len(document_chunks))],
                    documents=document_chunks,
                    metadatas=metadatas,
                )
                print(f"Loaded initial knowledge: {doc_name}")
            else:
                print(f"Failed to extract text from {item}")

    except Exception as e:
        print(f"An error occurred while loading initial knowledge: {e}")
