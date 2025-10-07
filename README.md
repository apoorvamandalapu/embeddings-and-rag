# Smart Knowledge Retrieval (RAG) Project

This project demonstrates **three RAG (Retrieval-Augmented Generation) pipelines** to answer questions from a knowledge base using embeddings and vector search.

---

## Project Overview

The project has three approaches:

1. **Normal RAG (In-Memory)**
   - Generates embeddings from `knowledge_base.csv` using **Google GenAI**.  
   - Stores embeddings in a `.pkl` file.  
   - Searches in-memory using **cosine similarity**.  
   - Script: `rag.py`

2. **RAG + ChromaDB**
   - Uses the same GenAI embeddings.  
   - Stores embeddings in **ChromaDB** for persistent, fast search.  
   - Script: `rag_db.py` (searches ChromaDB)  
   - Store script: `store.py`

3. **Vertex AI + ChromaDB RAG**
   - Generates embeddings using **Vertex AI**.  
   - Stores embeddings in **ChromaDB**.  
   - Searches and generates answers using `rag_service.py` and `main.py`.  
   - Store script: `store_vertex.py`

