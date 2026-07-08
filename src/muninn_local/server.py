import os
from mcp.server.fastmcp import FastMCP
from muninn_local.config import get_config
from muninn_local.embeddings import OllamaEmbedder
from muninn_local.chroma_store import ChromaStore
from muninn_local.project import get_project_id, get_project_name
from muninn_local.memory import MemoryManager

mcp = FastMCP("muninn")

_manager: MemoryManager | None = None


def _get_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        config = get_config()
        store = ChromaStore(config.chroma_dir)
        embedder = OllamaEmbedder(config.ollama_url, config.embed_model)
        project_id = get_project_id(os.getcwd())
        project_name = get_project_name(os.getcwd())
        _manager = MemoryManager(
            store=store,
            embedder=embedder,
            project_id=project_id,
            project_name=project_name,
        )
    return _manager


@mcp.tool()
async def memory_write(text: str, memory_type: str = "note", tags: str = "") -> str:
    """Write a memory entry to the current project's memory store.

    Use this tool to save important context, decisions, code patterns,
    or next-steps that should be persisted for future sessions.

    Args:
        text: The memory content to store (required).
        memory_type: Type of memory. Must be one of:
            summary, decision, next-steps, code-pattern, note (default).
        tags: Comma-separated tags for categorization (optional).
    """
    mgr = _get_manager()
    result = await mgr.write(text, memory_type=memory_type, tags=tags)
    short_id = result["id"][:8]
    return (
        f"✅ Memory saved — **{result['memory_type']}** · "
        f"`{short_id}…` · "
        f"project: `{result['project']}` · "
        f"tags: `{result['tags'] or '(none)'}`"
    )


@mcp.tool()
async def memory_search(query: str, top_k: int = 5) -> str:
    """Search the current project's memory store using semantic similarity.

    Embeds the query and returns the most relevant stored memories.

    Args:
        query: The search query text (required).
        top_k: Maximum number of results to return (default 5).
    """
    mgr = _get_manager()
    results = await mgr.search(query, top_k=top_k)
    if not results:
        return "No memories found."
    lines = ["## Search Results\n"]
    for i, r in enumerate(results, 1):
        dist_info = f" (distance: {r.get('distance'):.3f})" if "distance" in r else ""
        created = r.get("metadata", {}).get("created_at", "unknown")
        lines.append(f"### {i}. `{r['id'][:8]}…`{dist_info}")
        lines.append(f"{r['document']}")
        lines.append(f"_type: {r.get('metadata', {}).get('type', 'note')} · created: {created}_\n")
    return "\n".join(lines)


@mcp.tool()
async def memory_list(limit: int = 20, offset: int = 0) -> str:
    """List memories stored in the current project, ordered by creation time (newest first).

    Args:
        limit: Maximum number of memories to return (default 20).
        offset: Number of memories to skip for pagination (default 0).
    """
    mgr = _get_manager()
    results = await mgr.list_memories(limit=limit, offset=offset)
    if not results:
        return "No memories found."
    lines = ["## Memory List\n"]
    for i, r in enumerate(results, 1):
        created = r.get("metadata", {}).get("created_at", "unknown")
        mem_type = r.get("metadata", {}).get("type", "note")
        lines.append(f"### {i + offset}. `{r['id'][:8]}…` · **{mem_type}**")
        lines.append(f"{r['document']}")
        lines.append(f"_created: {created}_\n")
    return "\n".join(lines)


@mcp.tool()
async def memory_delete(memory_id: str) -> str:
    """Delete a specific memory from the current project by its ID.

    Args:
        memory_id: The full or partial ID of the memory to delete (required).
    """
    mgr = _get_manager()
    success = await mgr.delete(memory_id)
    if success:
        return f"✅ Deleted — `{memory_id}`"
    return f"❌ Not found — `{memory_id}`"


@mcp.tool()
async def global_memory_write(text: str, memory_type: str = "note", tags: str = "") -> str:
    """Write a memory entry to the global (cross-project) memory store.

    Global memories are shared across all projects and can be queried
    regardless of which project the MCP server is running in.

    Args:
        text: The memory content to store (required).
        memory_type: Type of memory. Must be one of:
            summary, decision, next-steps, code-pattern, note (default).
        tags: Comma-separated tags for categorization (optional).
    """
    mgr = _get_manager()
    result = await mgr.write(text, memory_type=memory_type, tags=tags, is_global=True)
    short_id = result["id"][:8]
    return (
        f"✅ Memory saved — **{result['memory_type']}** · "
        f"`{short_id}…` · "
        f"global · "
        f"origin: `{result['project']}`"
    )


@mcp.tool()
async def global_memory_search(query: str, top_k: int = 5) -> str:
    """Search the global (cross-project) memory store using semantic similarity.

    Embeds the query and returns the most relevant stored memories from all projects.

    Args:
        query: The search query text (required).
        top_k: Maximum number of results to return (default 5).
    """
    mgr = _get_manager()
    results = await mgr.search(query, top_k=top_k, is_global=True)
    if not results:
        return "No memories found."
    lines = ["## Global Search Results\n"]
    for i, r in enumerate(results, 1):
        dist_info = f" (distance: {r.get('distance'):.3f})" if "distance" in r else ""
        origin = r.get("metadata", {}).get("project_name", "unknown")
        created = r.get("metadata", {}).get("created_at", "unknown")
        lines.append(f"### {i}. `{r['id'][:8]}…`{dist_info}")
        lines.append(f"{r['document']}")
        lines.append(f"_origin: `{origin}` · type: {r.get('metadata', {}).get('type', 'note')} · created: {created}_\n")
    return "\n".join(lines)


@mcp.tool()
async def global_memory_list(limit: int = 20) -> str:
    """List globally stored memories, ordered by creation time (newest first).

    Args:
        limit: Maximum number of memories to return (default 20).
    """
    mgr = _get_manager()
    results = await mgr.list_memories(limit=limit, offset=0, is_global=True)
    if not results:
        return "No memories found."
    lines = ["## Global Memory List\n"]
    for i, r in enumerate(results, 1):
        origin = r.get("metadata", {}).get("project_name", "unknown")
        created = r.get("metadata", {}).get("created_at", "unknown")
        mem_type = r.get("metadata", {}).get("type", "note")
        lines.append(f"### {i}. `{r['id'][:8]}…` · **{mem_type}**")
        lines.append(f"{r['document']}")
        lines.append(f"_origin: `{origin}` · created: {created}_\n")
    return "\n".join(lines)
