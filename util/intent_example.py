from sentence_transformers import util
intent_examples = {
    "gaji": [
        "Berapa gaji saya bulan ini?",
        "Saya ingin tahu gaji saya.",
        "Gaji saya bulan ini berapa?",
        "Berapa upah saya?",
        "Bisakah saya melihat rincian gaji saya?"
    ],
    "cuti": [
        "Saya ingin mengajukan cuti.",
        "Cuti saya masih berapa hari?",
        "Bagaimana cara mengajukan cuti?",
        "Berapa sisa cuti saya?",
        "Apakah saya bisa mengambil cuti besok?"
    ]
}
def detect_intent(intent_model,intent_embeddings,user_query: str):
    query_embedding = intent_model.encode(user_query)
    max_similarity = 0
    detected_intent = None

    for intent, embeddings in intent_embeddings.items():
        similarities = util.cos_sim(query_embedding, embeddings).max().item()
        if similarities > max_similarity:
            max_similarity = similarities
            detected_intent = intent

    return detected_intent if max_similarity > 0.7 else None