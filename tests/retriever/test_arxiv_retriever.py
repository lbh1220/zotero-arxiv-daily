"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # The RSS fixture gives us paper IDs. The normal path calls
    # arxiv.Client().results(search) for richer metadata, so we mock the client
    # to keep the test offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author", affiliation=["Test University"])],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)
    assert all(p.abstract == "Test abstract" for p in papers)
    assert all(p.affiliations == ["Test University"] for p in papers)
    assert all(p.full_text is None for p in papers)


def test_arxiv_retriever_falls_back_to_rss_on_api_429(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)

    class FakeHTTPError(Exception):
        def __init__(self, status):
            super().__init__("Rate exceeded.")
            self.status = status

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            raise FakeHTTPError(429)

    monkeypatch.setattr(arxiv_retriever.arxiv, "HTTPError", FakeHTTPError)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == 2
    assert papers[0].title == "Neural Architecture Search for Efficient Transformers"
    assert papers[0].authors == ["Alice Smith", "Bob Jones"]
    assert papers[0].abstract == "We propose a neural architecture search method for efficient transformers."
    assert papers[0].url == "https://arxiv.org/abs/2508.14001"
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2508.14001"
    assert papers[0].affiliations is None
    assert all("Announce Type" not in p.abstract for p in papers)


def test_collect_author_affiliations_deduplicates_and_ignores_empty():
    authors = [
        SimpleNamespace(name="A", affiliation=["MIT", ""]),
        SimpleNamespace(name="B", affiliation=["MIT", "Stanford"]),
        SimpleNamespace(name="C", affiliation=[]),
    ]

    affiliations = arxiv_retriever._collect_author_affiliations(authors)

    assert affiliations == ["MIT", "Stanford"]


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
