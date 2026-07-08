import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from muninn_local.memory import MemoryManager


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.add = MagicMock()
    store.query = MagicMock()
    store.list_docs = MagicMock()
    store.delete = MagicMock()
    store.count = MagicMock(return_value=0)
    return store


@pytest.fixture
def mock_embedder():
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 768)
    embedder.embed_batch = AsyncMock(return_value=[[0.1] * 768])
    embedder.health_check = AsyncMock(return_value=True)
    return embedder


@pytest.fixture
def manager(mock_store, mock_embedder):
    return MemoryManager(
        store=mock_store,
        embedder=mock_embedder,
        project_id="abc123",
        project_name="test-project",
    )


class TestMemoryManagerInit:
    def test_init_stores_attributes(self, mock_store, mock_embedder):
        mgr = MemoryManager(
            store=mock_store,
            embedder=mock_embedder,
            project_id="proj1",
            project_name="MyProject",
        )
        assert mgr._project_id == "proj1"
        assert mgr._project_name == "MyProject"

    def test_project_collection_name(self, manager):
        assert manager._project_collection == "project_abc123"

    def test_global_collection_name(self, manager):
        assert manager._global_collection == "global"


class TestMemoryManagerWrite:
    async def test_write_project_memory_embeds_text(self, manager, mock_embedder):
        await manager.write("hello world")
        mock_embedder.embed.assert_awaited_once_with("hello world")

    async def test_write_project_memory_adds_to_store(self, manager, mock_store):
        result = await manager.write("test memory")
        mock_store.add.assert_called_once()

    async def test_write_project_memory_uses_project_collection(self, manager, mock_store):
        await manager.write("test")
        mock_store.add.assert_called_once()
        assert mock_store.add.call_args.kwargs["collection_name"] == "project_abc123"

    async def test_write_global_memory_uses_global_collection(self, manager, mock_store):
        await manager.write("global note", is_global=True)
        mock_store.add.assert_called_once()
        assert mock_store.add.call_args.kwargs["collection_name"] == "global"

    async def test_write_returns_dict_with_id(self, manager):
        result = await manager.write("test memory")
        assert isinstance(result, dict)
        assert "id" in result
        assert len(result["id"]) == 32

    async def test_write_stores_correct_metadata(self, manager, mock_store):
        await manager.write("test", memory_type="decision", tags="important,review")
        call_args = mock_store.add.call_args
        metadata = call_args[1]["metadata"]
        assert metadata["type"] == "decision"
        assert metadata["tags"] == "important,review"
        assert metadata["project_id"] == "abc123"
        assert metadata["project_name"] == "test-project"
        assert "created_at" in metadata

    async def test_write_global_memory_stores_project_context(self, manager, mock_store):
        await manager.write("global", is_global=True)
        call_args = mock_store.add.call_args
        metadata = call_args[1]["metadata"]
        assert metadata["project_id"] == "abc123"
        assert metadata["project_name"] == "test-project"


class TestMemoryManagerSearch:
    async def test_search_embeds_query(self, manager, mock_embedder):
        mock_embedder.embed.return_value = [0.5] * 768
        manager._store.query.return_value = {
            "ids": [["id1"]],
            "documents": [["result"]],
            "metadatas": [[{"type": "note"}]],
            "distances": [[0.1]],
        }
        await manager.search("find me")
        mock_embedder.embed.assert_awaited_once_with("find me")

    async def test_search_returns_list_of_dicts(self, manager):
        manager._store.query.return_value = {
            "ids": [["id1"]],
            "documents": [["result text"]],
            "metadatas": [[{"type": "note", "tags": "test", "created_at": "2026-01-01"}]],
            "distances": [[0.15]],
        }
        results = await manager.search("query")
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["id"] == "id1"
        assert results[0]["document"] == "result text"
        assert results[0]["distance"] == 0.15
        assert results[0]["metadata"]["type"] == "note"

    async def test_search_uses_project_collection_by_default(self, manager, mock_store):
        mock_store.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        await manager.search("test")
        mock_store.query.assert_called_once()
        assert mock_store.query.call_args.kwargs["collection_name"] == "project_abc123"

    async def test_search_global_uses_global_collection(self, manager, mock_store):
        mock_store.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        await manager.search("test", is_global=True)
        mock_store.query.assert_called_once()
        assert mock_store.query.call_args.kwargs["collection_name"] == "global"

    async def test_search_empty_results(self, manager):
        manager._store.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        results = await manager.search("nothing")
        assert results == []


class TestMemoryManagerList:
    async def test_list_returns_list_of_dicts(self, manager):
        manager._store.list_docs.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["doc1", "doc2"],
            "metadatas": [
                {"type": "note", "created_at": "2026-01-02"},
                {"type": "decision", "created_at": "2026-01-01"},
            ],
        }
        results = await manager.list_memories()
        assert isinstance(results, list)
        assert len(results) == 2

    async def test_list_passes_limit_and_offset(self, manager, mock_store):
        mock_store.list_docs.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }
        await manager.list_memories(limit=10, offset=5)
        mock_store.list_docs.assert_called_once()
        call_kwargs = mock_store.list_docs.call_args.kwargs
        assert call_kwargs["collection_name"] == "project_abc123"
        assert call_kwargs["limit"] == 10
        assert call_kwargs["offset"] == 5

    async def test_list_global_uses_global_collection(self, manager, mock_store):
        mock_store.list_docs.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }
        await manager.list_memories(is_global=True)
        mock_store.list_docs.assert_called_once()
        call_kwargs = mock_store.list_docs.call_args.kwargs
        assert call_kwargs["collection_name"] == "global"
        assert call_kwargs["limit"] == 20
        assert call_kwargs["offset"] == 0


class TestMemoryManagerDelete:
    async def test_delete_project_returns_true(self, manager, mock_store):
        mock_store.delete.return_value = True
        result = await manager.delete("some-id")
        assert result is True
        mock_store.delete.assert_called_once_with("project_abc123", "some-id")

    async def test_delete_global_returns_true(self, manager, mock_store):
        mock_store.delete.return_value = True
        result = await manager.delete("some-id", is_global=True)
        assert result is True
        mock_store.delete.assert_called_once_with("global", "some-id")

    async def test_delete_nonexistent_returns_false(self, manager, mock_store):
        mock_store.delete.return_value = False
        result = await manager.delete("missing")
        assert result is False
