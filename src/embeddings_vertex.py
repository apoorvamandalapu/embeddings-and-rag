import pandas as pd
from vertexai.language_models import TextEmbeddingModel
import vertexai
from dotenv import load_dotenv
import os

load_dotenv()



file_path = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.csv")

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
MODEL_NAME = "text-embedding-004"

vertexai.init(project=PROJECT_ID, location=LOCATION)

try:
    embedding_model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
    print(f"Vertex AI embedding model '{MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading Vertex AI model: {e}")
    print("Check project ID, location, and authentication.")
    exit()

def generate_embeddings(df):
    embeddings_list = []
    print("Generating embeddings with Vertex AI.")
    
    for i, text in enumerate(df["content"]):
        try:
            embeddings = embedding_model.get_embeddings([text])[0].values
            embeddings_list.append(embeddings)
        except Exception as e:
            print(f"Failed to get embedding for row {i}: {e}")
            embeddings_list.append(None) 
            continue

        if (i + 1) % 5 == 0:
            print(f"{i + 1}/{len(df)} rows done...")#checking 5 at time
            
    df["embedding"] = embeddings_list
    print("All embeddings created!")
    return df


if __name__ == "__main__":
    df = load_dataset()
    print(f"Loaded {len(df)} rows")
    df = generate_embeddings(df)
    df.dropna(subset=['embedding'], inplace=True)# Drop rows if embedding generation failed
    print(df.head())
    output_path = os.path.join(os.path.dirname(__file__), "data", "embeddings_vertex.pkl")
    df.to_pickle(output_path)
    print("Saved the Embeddings file")