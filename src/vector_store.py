# src/vector_store.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, persist_directory: str = "./data/chroma_db", collection_name: 
                 str = "knowledge_base"):
        self.client = chromadb.PersistentClient(path=persist_directory, settings=
                                                Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        ids = [f"{chunk['source']}_{chunk['chunk_id']}" for chunk in chunks]
        documents = [chunk['chunk_text'] for chunk in chunks]
        metadatas = [{"source": chunk['source'], "chunk_id": chunk['chunk_id']} 
                     for chunk in chunks]
        self.collection.add(ids=ids, embeddings=embeddings, documents=documents,
                             metadatas=metadatas)
        print(f"成功添加 {len(ids)} 个向量")

    def search(self, query: str, embedding_model, top_k: int = 5):
        query_embedding = embedding_model.embed(query).tolist()[0]
        results = self.collection.query(query_embeddings=[query_embedding], 
                                       n_results=top_k, include=["documents", "metadatas", "distances"])
        retrieved = []
        for i in range(len(results['ids'][0])):
            retrieved.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        return retrieved