import os
from pathlib import Path

import chromadb
from chromadb.config import Settings


class ChromaStore:
    def __init__(self, persist_dir: str):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collections: dict[str, chromadb.Collection] = {}

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    def add(
        self,
        collection_name: str,
        doc_id: str,
        document: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        coll = self.get_or_create_collection(collection_name)
        coll.add(
            ids=[doc_id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> dict:
        coll = self.get_or_create_collection(collection_name)
        n = min(top_k, coll.count())
        if n == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return coll.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

    def list_docs(
        self,
        collection_name: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        coll = self.get_or_create_collection(collection_name)
        total = coll.count()
        if total == 0 or offset >= total:
            return {"ids": [], "documents": [], "metadatas": []}
        result = coll.get(include=["documents", "metadatas"])
        ids = result["ids"]
        documents = result["documents"]
        metadatas = result["metadatas"]
        indices = list(range(len(ids)))
        indices.sort(
            key=lambda i: metadatas[i].get("created_at", "") if metadatas[i] else "",
            reverse=True,
        )
        ids = [ids[i] for i in indices]
        documents = [documents[i] for i in indices]
        metadatas = [metadatas[i] for i in indices]
        end = offset + limit
        return {
            "ids": ids[offset:end],
            "documents": documents[offset:end],
            "metadatas": metadatas[offset:end],
        }

    def delete(self, collection_name: str, doc_id: str) -> bool:
        coll = self.get_or_create_collection(collection_name)
        all_docs = coll.get(include=[])
        matched = [i for i in all_docs["ids"] if i.startswith(doc_id)]
        if not matched:
            return False
        coll.delete(ids=matched)
        return True

    def count(self, collection_name: str) -> int:
        coll = self.get_or_create_collection(collection_name)
        return coll.count()
