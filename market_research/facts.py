"""Pluggable, cited sources of facts about "my product".

Replaces sending one fixed sentence to a hosted GenAI agent and hoping it
somehow knows about the product. The default source reads product_narsi.md
directly and ranks its sections with BM25, offline, with no model call. A
knowledge-base-backed source is available behind config for anyone who
really has a DigitalOcean knowledge base set up.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from rank_bm25 import BM25Okapi

from market_research.categories import CATEGORIES
from market_research.errors import ConfigError
from market_research.models import Fact, ProductFacts

DEFAULT_PRODUCT_MARKDOWN = Path(__file__).resolve().parent.parent / "product_narsi.md"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class ProductFactsSource(Protocol):
    """Anything that can produce cited facts about "my product"."""

    def get_facts(self, product: str) -> ProductFacts: ...


class _Section:
    __slots__ = ("heading", "body")

    def __init__(self, heading: str, body: str) -> None:
        self.heading = heading
        self.body = body.strip()


def _parse_sections(markdown_text: str) -> list[_Section]:
    """Split markdown into sections by heading line (any '#' level)."""
    lines = markdown_text.splitlines()
    sections: list[_Section] = []
    current_heading = "Introduction"
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(_Section(current_heading, body))

    for line in lines:
        if line.startswith("#"):
            flush()
            current_heading = line.lstrip("#").strip() or current_heading
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return sections


class LocalMarkdownSource:
    """Default product facts source: BM25 over sections of a local markdown file.

    No model call, no network access; this runs purely from a file read, so
    importing or calling it in a test never reaches the network.
    """

    def __init__(self, markdown_path: Path | str = DEFAULT_PRODUCT_MARKDOWN, top_k: int = 2) -> None:
        self.markdown_path = Path(markdown_path)
        self.top_k = top_k
        if not self.markdown_path.exists():
            raise ConfigError(f"product markdown source not found: {self.markdown_path}")
        self._sections = _parse_sections(self.markdown_path.read_text(encoding="utf-8"))
        if not self._sections:
            raise ConfigError(f"product markdown source has no sections: {self.markdown_path}")
        self._bm25 = BM25Okapi([_tokenize(s.body) for s in self._sections])

    def _rank_for(self, query: str) -> list[_Section]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(scores, self._sections, strict=True), key=lambda pair: pair[0], reverse=True)
        return [section for _, section in ranked[: self.top_k]]

    def get_facts(self, product: str) -> ProductFacts:
        items: list[Fact] = []
        seen_headings: set[str] = set()
        for category in CATEGORIES:
            for section in self._rank_for(category):
                key = f"{category}:{section.heading}"
                if key in seen_headings:
                    continue
                seen_headings.add(key)
                snippet = section.body[:280]
                items.append(
                    Fact(
                        text=f"[{category}] {snippet}",
                        source=f"{self.markdown_path.name}#{section.heading}",
                        snippet=snippet,
                    )
                )
        return ProductFacts(product=product, items=items)

    def rank_sections(self, query: str) -> list[str]:
        """Expose ranked heading order for tests (e.g. "pricing" query)."""
        return [section.heading for section in self._rank_for(query)]


class DOKnowledgeBaseSource:
    """Optional source backed by a hosted DigitalOcean knowledge-base agent.

    Only used when FACTS_SOURCE=do_knowledge_base is configured; calls the
    hosted KB agent via the shared LLM provider layer and maps retrieval
    chunks to Fact.source. This performs real network I/O and is not
    exercised by the default test suite.
    """

    def __init__(self, llm, kb_agent_role: str = "knowledge_base") -> None:
        self._llm = llm
        self._kb_agent_role = kb_agent_role

    def get_facts(self, product: str) -> ProductFacts:
        items: list[Fact] = []
        for category in CATEGORIES:
            prompt = (
                f"Using only the attached knowledge base, retrieve facts about "
                f"{product} relevant to the category '{category}'. Cite the "
                f"document and section for every fact."
            )
            response = self._llm.call(prompt)
            text = str(response).strip()
            if not text:
                continue
            items.append(
                Fact(
                    text=f"[{category}] {text}",
                    source=f"knowledge_base:{product}:{category}",
                    snippet=text[:280],
                )
            )
        return ProductFacts(product=product, items=items)


def get_product_facts_source() -> ProductFactsSource:
    """Factory reading FACTS_SOURCE from the environment (default: local markdown)."""
    import os

    backend = os.getenv("FACTS_SOURCE", "local_markdown").strip().lower()
    if backend == "local_markdown":
        return LocalMarkdownSource()
    if backend == "do_knowledge_base":
        from market_research.llm import get_llm

        return DOKnowledgeBaseSource(get_llm("knowledge_base"))
    raise ConfigError(
        f"Unknown FACTS_SOURCE={backend!r}; expected local_markdown or do_knowledge_base"
    )
