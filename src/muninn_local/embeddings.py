import httpx


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
        )

    async def _post(self, payload: dict) -> dict:
        response = await self._client.post(
            f"{self._base_url}/api/embed", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def embed(self, text: str) -> list[float]:
        data = await self._post({"model": self._model, "input": text})
        return data["embeddings"][0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        data = await self._post({"model": self._model, "input": texts})
        return data["embeddings"]

    async def health_check(self) -> bool:
        try:
            await self._post({"model": self._model, "input": "health"})
            return True
        except Exception:
            return False