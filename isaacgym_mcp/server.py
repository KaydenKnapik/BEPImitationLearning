#!/usr/bin/env python3
"""MCP Server for Isaac Gym documentation at https://docs.robotsfan.com/isaacgym/"""

import asyncio
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

BASE_URL = "https://docs.robotsfan.com/isaacgym"

SITEMAP: dict[str, dict[str, str]] = {
    "index":                  {"title": "Isaac Gym Home",               "path": "index.html"},
    "about":                  {"title": "About Isaac Gym",              "path": "about.html"},
    "install":                {"title": "Installation",                 "path": "install.html"},
    "release_notes":          {"title": "Release Notes",                "path": "release_notes.html"},
    "faq":                    {"title": "FAQ",                          "path": "faq.html"},
    "examples/index":         {"title": "Examples",                     "path": "examples/index.html"},
    "examples/simple":        {"title": "Programming Examples",         "path": "examples/simple.html"},
    "examples/rl":            {"title": "Reinforcement Learning Examples","path": "examples/rl.html"},
    "examples/assets":        {"title": "Bundled Assets",               "path": "examples/assets.html"},
    "programming/index":      {"title": "Programming Guide",            "path": "programming/index.html"},
    "programming/simsetup":   {"title": "Simulation Setup",             "path": "programming/simsetup.html"},
    "programming/assets":     {"title": "Assets",                       "path": "programming/assets.html"},
    "programming/physics":    {"title": "Physics Simulation",           "path": "programming/physics.html"},
    "programming/tensors":    {"title": "Tensor API",                   "path": "programming/tensors.html"},
    "programming/forcesensors":{"title": "Force Sensors",               "path": "programming/forcesensors.html"},
    "programming/tuning":     {"title": "Simulation Tuning",            "path": "programming/tuning.html"},
    "programming/math":       {"title": "Math Utilities",               "path": "programming/math.html"},
    "programming/graphics":   {"title": "Graphics and Camera Sensors",  "path": "programming/graphics.html"},
    "programming/terrain":    {"title": "Terrains",                     "path": "programming/terrain.html"},
    "api/index":              {"title": "API Reference",                "path": "api/index.html"},
    "api/python/index":       {"title": "Python API",                   "path": "api/python/index.html"},
    "api/python/gym_py":      {"title": "Python Gym API",               "path": "api/python/gym_py.html"},
    "api/python/struct_py":   {"title": "Python Structures",            "path": "api/python/struct_py.html"},
    "api/python/enum_py":     {"title": "Python Enums",                 "path": "api/python/enum_py.html"},
    "api/python/const_py":    {"title": "Python Constants and Flags",   "path": "api/python/const_py.html"},
}

mcp = FastMCP("isaacgym_mcp")

_page_cache: dict[str, str] = {}


async def _fetch_page(path: str) -> str:
    """Fetch a doc page, strip boilerplate, return clean plain text."""
    if path in _page_cache:
        return _page_cache[path]

    url = f"{BASE_URL}/{path}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    main = (
        soup.find("div", {"class": "body"})
        or soup.find("div", {"role": "main"})
        or soup.find("main")
        or soup.find("article")
        or soup.body
    )
    text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    _page_cache[path] = text
    return text


def _path_to_title(path: str) -> str:
    return next((p["title"] for p in SITEMAP.values() if p["path"] == path), path)


# ── Tool 1: list sections ──────────────────────────────────────────────────────

@mcp.tool(
    name="isaacgym_list_sections",
    annotations={
        "title": "List Isaac Gym Documentation Sections",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def isaacgym_list_sections() -> str:
    """List all sections and pages in the Isaac Gym documentation.

    Returns an overview of the documentation structure with paths for every page.
    Use this to orient yourself before searching or fetching specific pages.

    Returns:
        str: Markdown list of all documentation sections with direct paths.
    """
    sections = {
        "Core": ["index", "about", "install", "release_notes", "faq"],
        "Examples": ["examples/index", "examples/simple", "examples/rl", "examples/assets"],
        "Programming": [
            "programming/index", "programming/simsetup", "programming/assets",
            "programming/physics", "programming/tensors", "programming/forcesensors",
            "programming/tuning", "programming/math", "programming/graphics", "programming/terrain",
        ],
        "API Reference": [
            "api/index", "api/python/index", "api/python/gym_py",
            "api/python/struct_py", "api/python/enum_py", "api/python/const_py",
        ],
    }

    lines = ["# Isaac Gym Documentation — All Pages", "", f"Base URL: {BASE_URL}/", ""]
    for section, keys in sections.items():
        lines.append(f"## {section}")
        for key in keys:
            page = SITEMAP[key]
            lines.append(f"- **{page['title']}**: `{page['path']}`")
        lines.append("")
    lines.append("Use `isaacgym_get_page` with a path to read a page, or `isaacgym_search` to find topics.")
    return "\n".join(lines)


# ── Tool 2: get page ───────────────────────────────────────────────────────────

class GetPageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    page_path: str = Field(
        ...,
        description=(
            "Relative path to the doc page, e.g. 'programming/tensors.html' or "
            "'api/python/gym_py.html'. Use isaacgym_list_sections to find valid paths."
        ),
        min_length=3,
    )


@mcp.tool(
    name="isaacgym_get_page",
    annotations={
        "title": "Get Isaac Gym Documentation Page",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def isaacgym_get_page(params: GetPageInput) -> str:
    """Fetch and return the text content of a specific Isaac Gym documentation page.

    Retrieves the page, strips navigation/boilerplate, and returns readable plain
    text with code blocks intact. Truncated to ~8000 characters. Use a section
    anchor suffix or a more specific path to jump to the right part.

    Args:
        params (GetPageInput):
            - page_path (str): Relative path, e.g. 'programming/tensors.html'

    Returns:
        str: Readable page content. Returns an error string if the page cannot be
             fetched or does not exist.
    """
    try:
        text = await _fetch_page(params.page_path)
        if len(text) > 8000:
            text = text[:8000] + "\n\n[Content truncated — use isaacgym_search with a more specific query to find a subsection.]"
        return text
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Error: Page '{params.page_path}' not found. Use isaacgym_list_sections to find valid paths."
        return f"Error: Failed to fetch page (HTTP {e.response.status_code})."
    except httpx.TimeoutException:
        return "Error: Request timed out. Please try again."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ── Tool 3: search ─────────────────────────────────────────────────────────────

class SearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(
        ...,
        description="Search terms, e.g. 'reward function', 'gym.create_env', 'tensor observation'",
        min_length=2,
        max_length=200,
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return (default 10, max 50)",
        ge=1,
        le=50,
    )


@mcp.tool(
    name="isaacgym_search",
    annotations={
        "title": "Search Isaac Gym Documentation",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def isaacgym_search(params: SearchInput) -> str:
    """Search the Isaac Gym documentation by keyword or phrase.

    Searches all documentation pages for the given terms. Returns a ranked list
    of matching pages with context snippets. Fetched pages are cached for speed
    on subsequent calls.

    Args:
        params (SearchInput):
            - query (str): Search terms, e.g. 'create_env', 'rigid body', 'camera sensor'
            - max_results (int): How many results to return (default 10, max 50)

    Returns:
        str: Markdown-formatted list of matching pages with relevance scores and
             context snippets. Returns "No results found." if nothing matches.
    """
    query_lower = params.query.lower()
    query_words = query_lower.split()

    paths = [page["path"] for page in SITEMAP.values()]
    fetched = await asyncio.gather(*[_fetch_page(p) for p in paths], return_exceptions=True)

    matches: list[tuple[int, str, str, str]] = []
    for path, content in zip(paths, fetched):
        if isinstance(content, Exception):
            continue
        content_lower = content.lower()

        word_hits = sum(1 for w in query_words if w in content_lower)
        phrase_hits = content_lower.count(query_lower)
        score = word_hits + phrase_hits * 3
        if score == 0:
            continue

        idx = content_lower.find(query_lower)
        if idx == -1:
            idx = next((content_lower.find(w) for w in query_words if w in content_lower), 0)
        start = max(0, idx - 100)
        end = min(len(content), idx + 250)
        snippet = " ".join(content[start:end].split())

        matches.append((score, _path_to_title(path), path, snippet))

    matches.sort(key=lambda x: x[0], reverse=True)
    matches = matches[: params.max_results]

    if not matches:
        return f"No results found for '{params.query}'."

    lines = [f"# Isaac Gym Docs — Search: '{params.query}'", "", f"{len(matches)} result(s):", ""]
    for score, title, path, snippet in matches:
        lines.append(f"## {title}")
        lines.append(f"- **Path**: `{path}`")
        lines.append(f"- **Score**: {score}")
        lines.append(f"- **Preview**: ...{snippet}...")
        lines.append("")
    lines.append("Use `isaacgym_get_page` with a path above to read the full content.")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
