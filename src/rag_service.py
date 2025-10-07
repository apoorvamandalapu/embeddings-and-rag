import chromadb
from google import genai
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv() 

try:
    _chroma_client = chromadb.PersistentClient(path="data/my_chroma_db_vertex")
    _collection = _chroma_client.get_collection(name="knowledge_base_vertex_collection")
    print("rag_service: Connected to ChromaDB collection.")
except Exception as e:
    print(f"rag_service: Error connecting to ChromaDB: {e}")
    print("Please ensure 'store_vertex.py' has been run successfully to create the database.")
    _collection = None 

def get_query_embedding(query):
    """Generates an embedding for a given query using the Gemini API."""
    if not query:
        return None
    
    api_key = os.getenv("GENAI_API_KEY")
    client = genai.Client(api_key) 

    response = client.models.embed_content(model="text-embedding-004", contents=query)
    return response.embeddings[0].values

def search_db(query, top_k=3):
    """
    Performs a similarity search on the ChromaDB collection.
    Returns the most relevant documents as a single string.
    """
    if _collection is None:
        return "" 

    q_emb = get_query_embedding(query)
    if q_emb is None:
        return ""
    
    results = _collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=['documents']
    )
    
    context = " ".join(results['documents'][0])
    return context

def generate_answer_from_context(query, context):
    """
    Generates a full answer using a large language model based on the provided context.
    """
    if not context:
        return "No relevant information found in the knowledge base."

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