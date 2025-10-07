import pandas as pd
import numpy as np
from google import genai
import os
from dotenv import load_dotenv

load_dotenv() 

df = pd.read_pickle("data/embeddings.pkl")
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_query_embedding(query):
    api_key = os.getenv("GENAI_API_KEY")
    client = genai.Client(api_key) 
    response = client.models.embed_content(model="gemini-embedding-001", contents=query)
    return response.embeddings[0].values

def search(query, top_k=3):
    q_emb = get_query_embedding(query)
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(q_emb, x))
    results = df.sort_values(by="similarity", ascending=False).head(top_k)
    return results

def generate_answer(query, context):
    api_key = os.getenv("GENAI_API_KEY")
    client = genai.Client(api_key) 
    prompt = f"Using the following context, answer the question: {query}\n\nContext: {context}"
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": prompt}]}]
    )
    return response.candidates[0].content.parts[0].text

if __name__ == "__main__":
    print("Search is ready. Type question or 'exit' to quit.")

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        print("Searching for relevant information...")
        results = search(query, top_k=3)  # top 3
        
        context = "\n".join(results["content"].tolist())
        
        print("\nGenerating a comprehensive answer...")
        answer = generate_answer(query, context)
        
        print("\nFinal Answer:")
        print(answer)