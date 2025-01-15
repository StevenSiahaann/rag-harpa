from typing import List, Dict
from util.prepare_document import *
def build_combined_prompt(query: str, context: List[str], history: List[Dict[str, str]]) -> str:
    base_prompt = """Kamu adalah HARPA Chatbot  yang akan membantu karyawan HARPA yang merupakan HARPA adalah solusi yang menyederhanakan proses HR dengan mengintegrasikan berbagai sumber data menjadi satu, menawarkan pengalaman pengguna responsif dengan keamanan dan segregasi data untuk fungsi sumber daya manusia perusahaan. Kamu adalah chatbot terpercaya dan juga bersahabat dengan para karyawan.

    
    
    You are an helpful assistant. I am going to ask you a question, and your answer should be based strictly on the history of our previous interactions and context provided of the document.
If the question is out of context of the document or cannot be answered based on the available context and history, make a best guess on the strictly context and history.say 
You may use your general knowledge to answer questions based on the context and history only. Do not use your general knowledge to answer anything out of context or history provided. If user says do not answer from the document, you are permitted to use general knowledge, otherwise Do not use your general knowledge.
Your response must be informative, concise and short provide as much relevant detail as possible, and never leave the question unanswered unless absolutely necessary.
        """
    #query = create_questions(query)
    user_prompt = f"Pertanyaannya adalah '{query}'. Berikut ini hal yang dapat kamu ketahui untuk menjawabnya :{context}. Jangan lupa untuk menjawab pertanyaan dengan bersahabat (contohnya : sertai emoji untuk setiap responnya dan gunakan kata - kata yang tidak kaku), serta kamu dapat menolak untuk menjawab secara sopan jika memang tidak mengetahuinya dan konteks yang diberikan tidak cukup untuk menjawabnya. Jika terdapat ajakan untuk bertegur sapa kamu dapat menjawab dengan perkenalan diri kamu, yaitu HARPA Chatbot."
    #print("USER PROMPT : ", user_prompt)
    #history_prompt = "\n".join([f"User: {item['query']}\nBot: {item['response']}" for item in history])
    history_prompt= "Here is all the history of our previous interactions you have:\n" + "\n".join([f"User: {item['query']}\nBot: {item['response']}" for item in history])
    history_prompt = clean_text(history_prompt)
    print()
    #print("HISTORY : ", f"{history_prompt}")
    print("COMBINED PROMPT : ", f"{base_prompt} \n HISTORY: {history_prompt} \n USER PROMPT: {user_prompt}")
    return f"{base_prompt} {history_prompt} {user_prompt}"
