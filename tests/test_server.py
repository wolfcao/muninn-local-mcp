import pytest
from unittest.mock import AsyncMock


class TestServerToolRegistration:
    def test_mcp_instance_has_seven_tools(self):
        import muninn_local.server as server_module

        mcp = server_module.mcp
        tool_names = [tool.name for tool in mcp._tool_manager._tools.values()]
        assert len(tool_names) == 7

    def test_all_expected_tool_names_present(self):
        import muninn_local.server as server_module

        mcp = server_module.mcp
        tool_names = {tool.name for tool in mcp._tool_manager._tools.values()}
        expected = {
            "memory_write",
            "memory_search",
            "memory_list",
            "memory_delete",
            "global_memory_write",
            "global_memory_search",
            "global_memory_list",
        }
        assert tool_names == expected


class TestServerToolSignatures:
    def test_memory_write_signature(self):
        import muninn_local.server as server_module

        mcp = server_module.mcp
        tools = {t.name: t for t in mcp._tool_manager._tools.values()}
        tool = tools["memory_write"]
        props = tool.parameters["properties"]
        assert "text" in props
        assert "memory_type" in props
        assert "tags" in props

    def test_memory_search_signature(self):
        import muninn_local.server as server_module

        mcp = server_module.mcp
        tools = {t.name: t for t in mcp._tool_manager._tools.values()}
        tool = tools["memory_search"]
        props = tool.parameters["properties"]
        assert "query" in props
        assert "top_k" in props

    def test_memory_list_signature(self):
        import muninn_local.server as server_module

        mcp = server_module.mcp
        tools = {t.name: t for t in mcp._tool_manager._tools.values()}
        tool = tools["memory_list"]
        props = tool.parameters["properties"]
        assert "limit" in props
        assert "offset" in props

    def test_memory_delete_signature(self):
        import muninn_local.server as server_module

        mcp = server_module.mcp
        tools = {t.name: t for t in mcp._tool_manager._tools.values()}
        tool = tools["memory_delete"]
        props = tool.parameters["properties"]
        assert "memory_id" in props

    def test_global_memory_write_signature(self):
        import muninn_local.server as server_module

        mcp = server_module.mcp
        tools = {t.name: t for t in mcp._tool_manager._tools.values()}
        tool = tools["global_memory_write"]
        props = tool.parameters["properties"]
        assert "text" in props
        assert "memory_type" in props
        assert "tags" in props

    def test_global_memory_search_signature(self):
        import muninn_local.server as server_module

        mcp = server_module.mcp
        tools = {t.name: t for t in mcp._tool_manager._tools.values()}
        tool = tools["global_memory_search"]
        props = tool.parameters["properties"]
        assert "query" in props
        assert "top_k" in props

    def test_global_memory_list_signature(self):
        import muninn_local.server as server_module

        mcp = server_module.mcp
        tools = {t.name: t for t in mcp._tool_manager._tools.values()}
        tool = tools["global_memory_list"]
        props = tool.parameters["properties"]
        assert "limit" in props


class TestServerToolDescriptions:
    def test_all_tools_have_descriptions(self):
        import muninn_local.server as server_module

        mcp = server_module.mcp
        for tool in mcp._tool_manager._tools.values():
            assert tool.description, f"Tool {tool.name} has no description"


class TestServerToolBehavior:
    async def test_memory_write_returns_markdown(self, monkeypatch):
        import muninn_local.server as server_module

        mock_mgr = AsyncMock()
        mock_mgr.write = AsyncMock(
            return_value={
                "id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
                "memory_type": "note",
                "tags": "test",
                "project": "test-p",
            }
        )
        monkeypatch.setattr(server_module, "_get_manager", lambda: mock_mgr)

        result = await server_module.memory_write("hello")

        assert "✅" in result
        assert "**note**" in result
        assert "a1b2c3d4" in result
        assert "test-p" in result

    async def test_memory_delete_success_markdown(self, monkeypatch):
        import muninn_local.server as server_module

        mock_mgr = AsyncMock()
        mock_mgr.delete = AsyncMock(return_value=True)
        monkeypatch.setattr(server_module, "_get_manager", lambda: mock_mgr)

        result = await server_module.memory_delete("abc123")

        assert "✅" in result
        assert "abc123" in result

    async def test_memory_delete_failure_markdown(self, monkeypatch):
        import muninn_local.server as server_module

        mock_mgr = AsyncMock()
        mock_mgr.delete = AsyncMock(return_value=False)
        monkeypatch.setattr(server_module, "_get_manager", lambda: mock_mgr)

        result = await server_module.memory_delete("missing")

        assert "❌" in result
        assert "missing" in result

    async def test_memory_search_empty_markdown(self, monkeypatch):
        import muninn_local.server as server_module

        mock_mgr = AsyncMock()
        mock_mgr.search = AsyncMock(return_value=[])
        monkeypatch.setattr(server_module, "_get_manager", lambda: mock_mgr)

        result = await server_module.memory_search("nothing")

        assert "No memories found" in result

    async def test_memory_search_results_markdown(self, monkeypatch):
        import muninn_local.server as server_module

        mock_mgr = AsyncMock()
        mock_mgr.search = AsyncMock(
            return_value=[
                {
                    "id": "id1",
                    "document": "test doc",
                    "metadata": {"type": "note", "tags": "test", "created_at": "2026-01-01"},
                    "distance": 0.15,
                }
            ]
        )
        monkeypatch.setattr(server_module, "_get_manager", lambda: mock_mgr)

        result = await server_module.memory_search("test")

        assert "test doc" in result
        assert "id1" in result

    async def test_memory_list_empty_markdown(self, monkeypatch):
        import muninn_local.server as server_module

        mock_mgr = AsyncMock()
        mock_mgr.list_memories = AsyncMock(return_value=[])
        monkeypatch.setattr(server_module, "_get_manager", lambda: mock_mgr)

        result = await server_module.memory_list()

        assert "No memories found" in result

    async def test_memory_list_results_markdown(self, monkeypatch):
        import muninn_local.server as server_module

        mock_mgr = AsyncMock()
        mock_mgr.list_memories = AsyncMock(
            return_value=[
                {"id": "id1", "document": "doc1", "metadata": {"type": "note"}}
            ]
        )
        monkeypatch.setattr(server_module, "_get_manager", lambda: mock_mgr)

        result = await server_module.memory_list()

        assert "doc1" in result
        assert "id1" in result

    async def test_global_write_uses_is_global(self, monkeypatch):
        import muninn_local.server as server_module

        mock_mgr = AsyncMock()
        mock_mgr.write = AsyncMock(
            return_value={
                "id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
                "memory_type": "note",
                "tags": "",
                "project": "test-p",
            }
        )
        monkeypatch.setattr(server_module, "_get_manager", lambda: mock_mgr)

        await server_module.global_memory_write("global text")

        mock_mgr.write.assert_called_once_with(
            "global text", memory_type="note", tags="", is_global=True
        )

    async def test_global_search_uses_is_global(self, monkeypatch):
        import muninn_local.server as server_module

        mock_mgr = AsyncMock()
        mock_mgr.search = AsyncMock(return_value=[])
        monkeypatch.setattr(server_module, "_get_manager", lambda: mock_mgr)

        await server_module.global_memory_search("query")

        mock_mgr.search.assert_called_once_with("query", top_k=5, is_global=True)

    async def test_global_list_uses_is_global(self, monkeypatch):
        import muninn_local.server as server_module

        mock_mgr = AsyncMock()
        mock_mgr.list_memories = AsyncMock(return_value=[])
        monkeypatch.setattr(server_module, "_get_manager", lambda: mock_mgr)

        await server_module.global_memory_list()

        mock_mgr.list_memories.assert_called_once_with(
            limit=20, offset=0, is_global=True
        )


class TestServerGetManager:
    def test_get_manager_returns_memory_manager(self):
        import muninn_local.server as server_module
        from muninn_local.memory import MemoryManager

        mgr = server_module._get_manager()
        assert isinstance(mgr, MemoryManager)

    def test_get_manager_is_singleton(self):
        import muninn_local.server as server_module

        mgr1 = server_module._get_manager()
        mgr2 = server_module._get_manager()
        assert mgr1 is mgr2
