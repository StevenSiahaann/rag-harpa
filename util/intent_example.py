from sentence_transformers import util
intent_examples = {
    "sisa cuti": [
        "Saya ingin mengajukan cuti.",
        "Cuti saya masih berapa hari?",
        "Berapa sisa cuti saya?",
        "Apakah saya bisa mengambil cuti besok?"
    ],
    "cuti terpakai":[
        "berapa cuti saya yang sudah saya pakai?",
        "berapa ijin yang sudah saya pakai?",
        "sudah berapa banyak saya menggunakan jatah cuti saya?",
        "sudah  berapa kali saya ijin?"
    ],
    "approval": [
        "Berapa pending approval saya?",
        "Berapa pengajuan saya yang masih pending?",
    ],
    "plafon":[
        "Berapa sisa plafon klaim saya?",
        "Cek sisa limit klaim saya",
        "Saya ingin tahu sisa plafon klaim",
        "Sisa plafon klaim saya berapa?",
        "Bisa tolong cek sisa limit klaim saya?",
        "Berapa batas klaim yang tersisa?",
        "Saya mau cek sisa plafon untuk klaim",
        "Cek limit klaim yang masih tersedia"
    ],
    "tentang HARPA":[
        "apa itu HARPA?",
        "apa keuntungan dari HARPA?",
        "dimana kantor HARPA?",
        "berapa nomor telpon HARPA?",
        "apa saja fitur utama dari HARPA?",
        "bagaimana penilaian customer terhadap HARPA?",
        "siapa Head of IT dari HARPA?",
        "bagaimana infrastructure aplikasi HARPA?"
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