"""Tests for multi-profile batch execution."""

import email

import pytest
from omegaconf import open_dict

from tests.canned_responses import make_sample_paper, make_stub_openai_client, make_stub_smtp, make_stub_zotero_client
from zotero_arxiv_daily.batch_executor import BatchExecutor


def _extract_subject(raw_message: str) -> str:
    message = email.message_from_string(raw_message)
    return str(email.header.make_header(email.header.decode_header(message["Subject"])))


def test_batch_executor_sends_one_email_per_profile(config, monkeypatch):
    import smtplib

    with open_dict(config):
        config.profiles = [
            {
                "name": "Group A",
                "include_path": ["survey", "survey/**"],
            },
            {
                "name": "Group B",
                "include_path": ["survey/topic-a", "survey/topic-a/**"],
                "max_paper_num": 1,
            },
        ]

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self: [
            make_sample_paper(title="Paper 1", url="https://arxiv.org/abs/1", pdf_url="https://arxiv.org/pdf/1"),
            make_sample_paper(title="Paper 2", url="https://arxiv.org/abs/2", pdf_url="https://arxiv.org/pdf/2"),
        ],
    )
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    executor = BatchExecutor(config)
    executor.run()

    assert len(sent) == 2
    subjects = [_extract_subject(body) for _, _, body in sent]
    assert any("Group A" in subject for subject in subjects)
    assert any("Group B" in subject for subject in subjects)


def test_batch_executor_continues_after_profile_failure_and_raises_at_end(config, monkeypatch):
    import smtplib

    with open_dict(config):
        config.profiles = [
            {
                "name": "Good 1",
                "include_path": ["survey", "survey/**"],
            },
            {
                "name": "Bad",
                "include_path": ["missing-folder", "missing-folder/**"],
            },
            {
                "name": "Good 2",
                "include_path": ["survey/topic-a", "survey/topic-a/**"],
            },
        ]

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self: [make_sample_paper(title="Paper 1")],
    )
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    executor = BatchExecutor(config)
    with pytest.raises(RuntimeError, match="One or more profiles failed"):
        executor.run()

    assert len(sent) == 2
    subjects = [_extract_subject(body) for _, _, body in sent]
    assert any("Good 1" in subject for subject in subjects)
    assert any("Good 2" in subject for subject in subjects)
    assert all("Bad" not in subject for subject in subjects)
