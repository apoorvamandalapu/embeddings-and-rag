import pandas as pd
import chromadb


try:
    df = pd.read_pickle("data/embeddings_vertex.pkl")
    print("Vertex AI embeddings loaded successfully.")
except FileNotFoundError:
    print("Error: Pkl is not found.")
    exit()

chroma_client = chromadb.PersistentClient(path="data/my_chroma_db_vertex")

collection_name = "knowledge_base_vertex_collection"

collection = chroma_client.get_or_create_collection(
    name=collection_name
)
print(f"Collection '{collection_name}' is ready.")

documents = df['content'].tolist()
embeddings = df['embedding'].tolist()
metadatas = df[['title', 'category']].to_dict('records')
ids = [f"vertex_id_{i}" for i in range(len(df))]

print(f"Adding {len(documents)} documents to the collection...")

collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print("All documents added to ChromaDB.")
print(f"Total documents in collection: {collection.count()}")