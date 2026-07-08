import pytest
import httpx
import respx
from muninn_local.embeddings import OllamaEmbedder


@pytest.fixture
def embedder():
    return OllamaEmbedder(
        base_url="http://localhost:11434", model="mxbai-embed-large"
    )


@pytest.fixture
def mock_ollama_embed():
    with respx.mock as mock:
        mock.post("http://localhost:11434/api/embed").respond(
            json={"embeddings": [[0.1] * 1024]}
        )
        yield mock


class TestOllamaEmbedder:
    async def test_embed_returns_1024_dim_vector(self, embedder, mock_ollama_embed):
        result = await embedder.embed("hello world")
        assert len(result) == 1024
        assert isinstance(result, list)
        assert isinstance(result[0], float)

    async def test_embed_batch_returns_correct_count(self, embedder):
        with respx.mock as mock:
            mock.post("http://localhost:11434/api/embed").respond(
                json={"embeddings": [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]}
            )
            results = await embedder.embed_batch(["a", "b", "c"])
            assert len(results) == 3
            assert all(len(v) == 1024 for v in results)

    async def test_embed_batch_single_item(self, embedder):
        with respx.mock as mock:
            mock.post("http://localhost:11434/api/embed").respond(
                json={"embeddings": [[0.5] * 1024]}
            )
            results = await embedder.embed_batch(["solo"])
            assert len(results) == 1
            assert len(results[0]) == 1024

    async def test_embed_empty_string_handled(self, embedder):
        with respx.mock as mock:
            mock.post("http://localhost:11434/api/embed").respond(
                json={"embeddings": [[0.0] * 1024]}
            )
            result = await embedder.embed("")
            assert len(result) == 1024

    async def test_ollama_unreachable_raises(self, embedder):
        with respx.mock as mock:
            mock.post("http://localhost:11434/api/embed").mock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            with pytest.raises(Exception) as exc_info:
                await embedder.embed("test")
            assert "Connection refused" in str(exc_info.value)

    async def test_health_check_ok(self, embedder):
        with respx.mock as mock:
            mock.post("http://localhost:11434/api/embed").respond(
                json={"embeddings": [[0.0] * 1024]}
            )
            result = await embedder.health_check()
            assert result is True

    async def test_health_check_fail(self, embedder):
        with respx.mock as mock:
            mock.post("http://localhost:11434/api/embed").mock(
                side_effect=httpx.ConnectError("no route to host")
            )
            result = await embedder.health_check()
            assert result is False