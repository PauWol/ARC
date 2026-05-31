"""
Web search utility functions for LLM agents.
Three versions ranging from zero-dependency to full-featured.

Dependencies per version:
  v1 (DuckDuckGo, no key): pip install requests beautifulsoup4
  v2 (SerpAPI / Brave):    pip install requests
  v3 (agent-ready):        pip install requests beautifulsoup4 (+ optional openai/anthropic)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus, urlencode

import requests
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────────────────────────────────────
# Shared data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    position: int = 0
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "position": self.position,
            "source": self.source,
        }

    def to_context_string(self) -> str:
        """Compact format for stuffing into an LLM prompt."""
        return f"[{self.position}] {self.title}\n{self.url}\n{self.snippet}"


# ─────────────────────────────────────────────────────────────────────────────
# VERSION 1 — DuckDuckGo HTML scrape (no API key required)
# ─────────────────────────────────────────────────────────────────────────────

def search_duckduckgo(
    query: str,
    max_results: int = 8,
    timeout: int = 10,
    safe_search: bool = True,
) -> list[SearchResult]:
    """
    Scrape DuckDuckGo HTML results. No API key needed.

    Pros:  free, zero setup, good for prototyping
    Cons:  fragile (markup can change), rate-limited, no structured metadata

    Args:
        query:       Search query string.
        max_results: Maximum number of results to return (1–20).
        timeout:     HTTP timeout in seconds.
        safe_search: Enable DDG safe-search filter.

    Returns:
        List of SearchResult objects, ordered by relevance.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    params = {
        "q": query,
        "kl": "us-en",
        "kp": "1" if safe_search else "-1",
    }
    url = f"https://html.duckduckgo.com/html/?{urlencode(params)}"

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[SearchResult] = []

    for i, r in enumerate(soup.select(".result__body")[:max_results], start=1):
        title_tag = r.select_one(".result__title a")
        snippet_tag = r.select_one(".result__snippet")

        if not title_tag:
            continue

        raw_href = title_tag.get("href", "")
        # DDG wraps URLs — extract the real one
        url_match = re.search(r"uddg=([^&]+)", raw_href)
        clean_url = (
            requests.utils.unquote(url_match.group(1))
            if url_match
            else raw_href
        )

        results.append(
            SearchResult(
                title=title_tag.get_text(strip=True),
                url=clean_url,
                snippet=snippet_tag.get_text(strip=True) if snippet_tag else "",
                position=i,
                source="duckduckgo",
            )
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# VERSION 2 — Production-grade with multiple API back-ends
# ─────────────────────────────────────────────────────────────────────────────

class WebSearcher:
    """
    Production-ready searcher supporting Brave Search API, SerpAPI, and
    Google Custom Search as interchangeable back-ends.

    Usage:
        searcher = WebSearcher(provider="brave", api_key="YOUR_KEY")
        results  = searcher.search("best python web frameworks 2025", n=5)

    Brave Search API:     https://api.search.brave.com  (free tier: 2k req/month)
    SerpAPI:              https://serpapi.com            (free tier: 100 req/month)
    Google Custom Search: https://developers.google.com/custom-search
    """

    PROVIDERS = ("brave", "serpapi", "google")

    def __init__(
        self,
        provider: str = "brave",
        api_key: str | None = None,
        google_cx: str | None = None,
        timeout: int = 15,
        retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        if provider not in self.PROVIDERS:
            raise ValueError(f"provider must be one of {self.PROVIDERS}")

        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY", "")
        self.google_cx = google_cx or os.getenv("GOOGLE_CX", "")
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self._session = requests.Session()

    # ── public interface ──────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n: int = 8,
        country: str = "us",
        language: str = "en",
    ) -> list[SearchResult]:
        """Run a web search and return up to *n* results."""
        dispatch = {
            "brave":   self._brave_search,
            "serpapi": self._serpapi_search,
            "google":  self._google_search,
        }
        for attempt in range(self.retries + 1):
            try:
                return dispatch[self.provider](query, n, country, language)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise
        return []

    def search_and_format(self, query: str, n: int = 5) -> str:
        """Return results as a plain-text block ready for an LLM prompt."""
        results = self.search(query, n=n)
        if not results:
            return f"No results found for: {query}"
        lines = [f"Search results for: {query}\n"]
        lines += [r.to_context_string() for r in results]
        return "\n\n".join(lines)

    # ── back-end implementations ──────────────────────────────────────────

    def _brave_search(
        self, query: str, n: int, country: str, language: str
    ) -> list[SearchResult]:
        resp = self._session.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept":               "application/json",
                "Accept-Encoding":      "gzip",
                "X-Subscription-Token": self.api_key,
            },
            params={
                "q":       query,
                "count":   min(n, 20),
                "country": country,
                "search_lang": language,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("description", ""),
                position=i,
                source="brave",
            )
            for i, r in enumerate(
                data.get("web", {}).get("results", [])[:n], start=1
            )
        ]

    def _serpapi_search(
        self, query: str, n: int, country: str, language: str
    ) -> list[SearchResult]:
        resp = self._session.get(
            "https://serpapi.com/search",
            params={
                "q":       query,
                "api_key": self.api_key,
                "num":     min(n, 10),
                "gl":      country,
                "hl":      language,
                "engine":  "google",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
                position=r.get("position", i),
                source="serpapi",
            )
            for i, r in enumerate(
                data.get("organic_results", [])[:n], start=1
            )
        ]

    def _google_search(
        self, query: str, n: int, country: str, language: str
    ) -> list[SearchResult]:
        resp = self._session.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "q":   query,
                "key": self.api_key,
                "cx":  self.google_cx,
                "num": min(n, 10),
                "gl":  country,
                "lr":  f"lang_{language}",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
                position=i,
                source="google",
            )
            for i, r in enumerate(data.get("items", [])[:n], start=1)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# VERSION 3 — Agent tool definitions (Anthropic + OpenAI format)
# ─────────────────────────────────────────────────────────────────────────────

# ── Tool schemas ──────────────────────────────────────────────────────────────

ANTHROPIC_SEARCH_TOOL: dict[str, Any] = {
    "name": "web_search",
    "description": (
        "Search the web for current information. Use this when you need "
        "up-to-date facts, recent events, or anything beyond your training data. "
        "Returns a list of relevant web results with titles, URLs, and snippets."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific and concise.",
            },
            "n": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 10).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

OPENAI_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": ANTHROPIC_SEARCH_TOOL["description"],
        "parameters": ANTHROPIC_SEARCH_TOOL["input_schema"],
    },
}


# ── Tool executor ─────────────────────────────────────────────────────────────

@dataclass
class AgentSearchTool:
    """
    Drop-in tool executor for agentic loops.

    Wraps any back-end (v1 scrape or v2 API) and exposes:
      - anthropic_tool_def  → pass to `tools=` in Anthropic API calls
      - openai_tool_def     → pass to `tools=` in OpenAI API calls
      - execute(input_dict) → call this when the model requests the tool

    Example (Anthropic agentic loop):

        tool = AgentSearchTool(provider="brave", api_key="sk-...")

        response = client.messages.create(
            model="claude-opus-4-5",
            tools=[tool.anthropic_tool_def],
            messages=[{"role": "user", "content": "What won best picture in 2025?"}],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "web_search":
                result = tool.execute(block.input)
                # feed result back as tool_result ...
    """

    provider: str = "duckduckgo"
    api_key: str | None = None
    google_cx: str | None = None
    max_results_cap: int = 10
    _searcher: WebSearcher | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.provider != "duckduckgo":
            self._searcher = WebSearcher(
                provider=self.provider,
                api_key=self.api_key,
                google_cx=self.google_cx,
            )

    @property
    def anthropic_tool_def(self) -> dict[str, Any]:
        return ANTHROPIC_SEARCH_TOOL

    @property
    def openai_tool_def(self) -> dict[str, Any]:
        return OPENAI_SEARCH_TOOL

    def execute(self, tool_input: dict[str, Any]) -> str:
        """
        Run the search and return a JSON string (suitable as tool_result content).

        Args:
            tool_input: Dict with keys 'query' (required) and 'n' (optional).

        Returns:
            JSON string: {"results": [...], "query": "...", "count": N}
        """
        query = tool_input.get("query", "").strip()
        n = min(int(tool_input.get("n", 5)), self.max_results_cap)

        if not query:
            return json.dumps({"error": "query must not be empty"})

        try:
            if self.provider == "duckduckgo":
                results = search_duckduckgo(query, max_results=n)
            else:
                results = self._searcher.search(query, n=n)  # type: ignore[union-attr]
        except Exception as exc:
            return json.dumps({"error": str(exc), "query": query})

        return json.dumps(
            {
                "query": query,
                "count": len(results),
                "results": [r.to_dict() for r in results],
            },
            ensure_ascii=False,
            indent=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Optional: fetch and clean a page's full text (for follow-up reads)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_page_text(url: str, timeout: int = 12, max_chars: int = 8_000) -> str:
    """
    Fetch a URL and return clean plain text (no HTML noise).
    Useful when an agent wants to read a full article after a search.

    Args:
        url:       Target URL.
        timeout:   HTTP timeout in seconds.
        max_chars: Truncate output to this many characters (avoids context overflow).

    Returns:
        Extracted plain text, truncated to max_chars.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove boilerplate tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== v1: DuckDuckGo scrape ===")
    for r in search_duckduckgo("anthropic claude 2025", max_results=3):
        print(r.to_context_string(), "\n")

    print("\n=== v3: AgentSearchTool (duckduckgo) ===")
    tool = AgentSearchTool(provider="duckduckgo")
    output = tool.execute({"query": "latest AI news", "n": 3})
    print(output)

    # To use a real API back-end, uncomment and set your key:
    # print("\n=== v2: Brave Search ===")
    # searcher = WebSearcher(provider="brave", api_key="YOUR_BRAVE_KEY")
    # print(searcher.search_and_format("python async best practices", n=4))