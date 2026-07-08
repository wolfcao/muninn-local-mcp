import uuid
from datetime import datetime, timezone

from muninn_local.chroma_store import ChromaStore
from muninn_local.embeddings import OllamaEmbedder


class MemoryManager:
    def __init__(self, store: ChromaStore, embedder: OllamaEmbedder, project_id: str, project_name: str):
        self._store = store
        self._embedder = embedder
        self._project_id = project_id
        self._project_name = project_name
        self._project_collection = f"project_{project_id}"
        self._global_collection = "global"

    def _collection(self, is_global: bool) -> str:
        return self._global_collection if is_global else self._project_collection

    async def write(self, text: str, memory_type: str = "note", tags: str = "", *, is_global: bool = False) -> dict:
        memory_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        embedding = await self._embedder.embed(text)

        collection = self._collection(is_global)
        metadata = {
            "type": memory_type,
            "tags": tags,
            "project_id": self._project_id,
            "project_name": self._project_name,
            "created_at": created_at,
        }

        self._store.add(
            collection_name=collection,
            doc_id=memory_id,
            document=text,
            embedding=embedding,
            metadata=metadata,
        )

        return {
            "id": memory_id,
            "text": text,
            "memory_type": memory_type,
            "tags": tags,
            "project": self._project_name,
            "created_at": created_at,
            "is_global": is_global,
        }

    async def search(self, query: str, top_k: int = 5, *, is_global: bool = False) -> list[dict]:
        query_embedding = await self._embedder.embed(query)
        collection = self._collection(is_global)

        result = self._store.query(
            collection_name=collection,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        results = []
        for i in range(len(ids)):
            results.append({
                "id": ids[i],
                "document": documents[i] if documents[i] else "",
                "distance": distances[i],
                "metadata": metadatas[i] if metadatas[i] else {},
            })

        return results

    async def list_memories(self, limit: int = 20, offset: int = 0, *, is_global: bool = False) -> list[dict]:
        collection = self._collection(is_global)
        result = self._store.list_docs(
            collection_name=collection,
            limit=limit,
            offset=offset,
        )

        ids = result["ids"]
        documents = result["documents"]
        metadatas = result["metadatas"]

        results = []
        for i in range(len(ids)):
            results.append({
                "id": ids[i],
                "document": documents[i] if documents[i] else "",
                "metadata": metadatas[i] if metadatas[i] else {},
            })

        return results

    async def delete(self, memory_id: str, *, is_global: bool = False) -> bool:
        collection = self._collection(is_global)
        return self._store.delete(collection, memory_id)