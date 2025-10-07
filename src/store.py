import pandas as pd
import chromadb

try:
    df = pd.read_pickle("data/embeddings.pkl")
    print("Saved the Embeddings file.")
except FileNotFoundError:
    print("Error: Pkl is not found.")
    exit()

chroma_client = chromadb.PersistentClient(path="data/my_chroma_db")
collection_name = "knowledge_base_collection"

collection = chroma_client.get_or_create_collection(
    name=collection_name
)
print(f"Collection '{collection_name}' is ready.")

documents = df['content'].tolist()
embeddings = df['embedding'].tolist()
metadatas = df[['title', 'category']].to_dict('records')
ids = [f"id_{i}" for i in range(len(df))]

print(f"Adding {len(documents)} documents to the collection...")

collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print("All documents added.")
print(f"Total documents in collection: {collection.count()}")