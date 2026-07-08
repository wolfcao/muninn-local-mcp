import pytest
from muninn_local.chroma_store import ChromaStore


@pytest.fixture
def chroma_store(tmp_path):
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    store = ChromaStore(str(persist_dir))
    yield store


class TestChromaStoreBasic:
    def test_init_creates_persist_dir(self, tmp_path):
        persist_dir = tmp_path / "new_store"
        store = ChromaStore(str(persist_dir))
        assert persist_dir.exists()

    def test_get_or_create_collection(self, chroma_store):
        coll = chroma_store.get_or_create_collection("test_coll")
        assert coll is not None
        assert coll.name == "test_coll"

    def test_get_or_create_collection_idempotent(self, chroma_store):
        coll1 = chroma_store.get_or_create_collection("test_coll")
        coll2 = chroma_store.get_or_create_collection("test_coll")
        assert coll1.name == coll2.name


class TestChromaStoreAddAndQuery:
    def test_add_and_query_end_to_end(self, chroma_store):
        chroma_store.add(
            collection_name="memories",
            doc_id="doc1",
            document="Remember the milk",
            embedding=[0.1, 0.2, 0.3],
            metadata={"created_at": "2026-01-01T00:00:00", "type": "note"},
        )

        result = chroma_store.query(
            collection_name="memories",
            query_embedding=[0.1, 0.2, 0.3],
            top_k=3,
        )

        assert "ids" in result
        assert result["ids"] == [["doc1"]]
        assert result["documents"] == [["Remember the milk"]]
        assert result["metadatas"] == [[{"created_at": "2026-01-01T00:00:00", "type": "note"}]]

    def test_query_respects_top_k(self, chroma_store):
        for i in range(5):
            chroma_store.add(
                collection_name="memories",
                doc_id=f"doc{i}",
                document=f"memory {i}",
                embedding=[0.1 * (i + 1), 0.2, 0.3],
                metadata={"created_at": f"2026-01-0{i+1}T00:00:00"},
            )

        result = chroma_store.query(
            collection_name="memories",
            query_embedding=[0.1, 0.2, 0.3],
            top_k=2,
        )

        assert len(result["ids"][0]) == 2


class TestChromaStoreDelete:
    def test_delete_existing_id(self, chroma_store):
        chroma_store.add(
            collection_name="memories",
            doc_id="doc1",
            document="test",
            embedding=[0.1, 0.2, 0.3],
            metadata={"created_at": "2026-01-01T00:00:00"},
        )

        result = chroma_store.delete("memories", "doc1")
        assert result is True
        assert chroma_store.count("memories") == 0

    def test_delete_nonexistent_id(self, chroma_store):
        chroma_store.get_or_create_collection("memories")
        result = chroma_store.delete("memories", "nonexistent")
        assert result is False


class TestChromaStoreListDocs:
    def test_list_docs_returns_documents(self, chroma_store):
        chroma_store.add(
            collection_name="memories",
            doc_id="doc1",
            document="first",
            embedding=[0.1, 0.2, 0.3],
            metadata={"created_at": "2026-01-01T00:00:00"},
        )
        chroma_store.add(
            collection_name="memories",
            doc_id="doc2",
            document="second",
            embedding=[0.1, 0.2, 0.3],
            metadata={"created_at": "2026-01-02T00:00:00"},
        )

        result = chroma_store.list_docs("memories", limit=10, offset=0)

        assert "ids" in result
        assert len(result["ids"]) == 2

    def test_list_docs_sorted_by_created_at_desc(self, chroma_store):
        chroma_store.add(
            collection_name="memories",
            doc_id="doc_old",
            document="older",
            embedding=[0.1, 0.2, 0.3],
            metadata={"created_at": "2026-01-01T00:00:00"},
        )
        chroma_store.add(
            collection_name="memories",
            doc_id="doc_new",
            document="newer",
            embedding=[0.1, 0.2, 0.3],
            metadata={"created_at": "2026-01-03T00:00:00"},
        )

        result = chroma_store.list_docs("memories", limit=10, offset=0)

        ids = result["ids"]
        assert ids[0] == "doc_new"
        assert ids[1] == "doc_old"

    def test_list_docs_pagination(self, chroma_store):
        for i in range(5):
            chroma_store.add(
                collection_name="memories",
                doc_id=f"doc{i}",
                document=f"memory {i}",
                embedding=[0.1, 0.2, 0.3],
                metadata={"created_at": f"2026-01-0{i+1}T00:00:00"},
            )

        page1 = chroma_store.list_docs("memories", limit=3, offset=0)
        page2 = chroma_store.list_docs("memories", limit=3, offset=3)

        assert len(page1["ids"]) == 3
        assert len(page2["ids"]) == 2

    def test_list_docs_empty_collection(self, chroma_store):
        chroma_store.get_or_create_collection("memories")
        result = chroma_store.list_docs("memories")
        assert len(result["ids"]) == 0


class TestChromaStoreCount:
    def test_count_returns_zero_for_empty(self, chroma_store):
        chroma_store.get_or_create_collection("memories")
        assert chroma_store.count("memories") == 0

    def test_count_accurate_after_adds(self, chroma_store):
        for i in range(3):
            chroma_store.add(
                collection_name="memories",
                doc_id=f"doc{i}",
                document=f"memory {i}",
                embedding=[0.1, 0.2, 0.3],
                metadata={"created_at": f"2026-01-0{i+1}T00:00:00"},
            )

        assert chroma_store.count("memories") == 3

    def test_count_updates_after_delete(self, chroma_store):
        chroma_store.add(
            collection_name="memories",
            doc_id="doc1",
            document="test",
            embedding=[0.1, 0.2, 0.3],
            metadata={"created_at": "2026-01-01T00:00:00"},
        )
        assert chroma_store.count("memories") == 1

        chroma_store.delete("memories", "doc1")
        assert chroma_store.count("memories") == 0
