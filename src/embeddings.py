import pandas as pd
from google import genai
import os
from dotenv import load_dotenv

load_dotenv() 

file_path = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.csv")

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def generate_embeddings(df):
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise ValueError("GENAI_API_KEY environment variable not set!")

    client = genai.Client(api_key="")  
    embeddings_list = []

    print("Generating embeddings...")
    for i, text in enumerate(df["content"]):
        response = client.models.embed_content(model="gemini-embedding-001", contents=text)
        embeddings_list.append(response.embeddings[0].values)
        if (i + 1) % 5 == 0: #checking 5 at time
            print(f"{i + 1}/{len(df)} rows done...")

    df["embedding"] = embeddings_list
    print("All embeddings created!")
    return df

if __name__ == "__main__":
    df = load_dataset()             
    print(f"Loaded {len(df)} rows")
    df = generate_embeddings(df)     
    print(df.head()) 
    output_path = os.path.join(os.path.dirname(__file__), "data", "embeddings.pkl")
    df.to_pickle(output_path)    
    print("Saved the Embeddings file")