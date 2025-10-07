import pandas as pd
import numpy as np
from google import genai
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

try:
    chroma_client = chromadb.PersistentClient(path="data/my_chroma_db")
    collection = chroma_client.get_collection(name="knowledge_base_collection")
    print("Connected to ChromaDB collection.")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    print("Please ensure 'store.py' has been run successfully to create the database.")
    exit()

def get_query_embedding(query):
    api_key = os.getenv("GENAI_API_KEY")
    client = genai.Client(api_key)  
    response = client.models.embed_content(model="gemini-embedding-001", contents=query)
    return response.embeddings[0].values

def search(query, top_k=3):
    q_emb = get_query_embedding(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=['documents']
    )
    return results

def generate_answer(query, context):
    api_key = os.getenv("GENAI_API_KEY")
    client = genai.Client(api_key) 
    prompt = f"""
    You are a helpful assistant. Use only the following information to answer the question.
    If the answer is not in the provided information, say "I don't have enough information to answer that."

    Question: {query}

    Context:
    {context}
    """
    
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
        
        try:
            print("Searching for relevant information...")
            results = search(query, top_k=3)  
            
            context = "\n".join(results['documents'][0])
            if not context.strip():
                print("No relevant information found in the knowledge base.")
                continue

            print("\nGenerating a comprehensive answer...")
            answer = generate_answer(query, context)
            
            print("\nFinal Answer:")
            print(answer)
        
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please check your API keys and code.")