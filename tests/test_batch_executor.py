"""Tests for multi-profile batch execution."""

import email

import pytest
from omegaconf import open_dict

from tests.canned_responses import make_sample_paper, make_stub_openai_client, make_stub_smtp, make_stub_zotero_client
from zotero_arxiv_daily.batch_executor import BatchExecutor, merge_profile_results


@pytest.fixture(autouse=True)
def no_profile_send_delay(config):
    with open_dict(config):
        config.email.profile_send_delay_seconds = 0
        config.email.merge_profiles = False


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


def test_batch_executor_waits_between_profile_emails(config, monkeypatch):
    import smtplib

    with open_dict(config):
        config.email.profile_send_delay_seconds = 2
        config.profiles = [
            {
                "name": "Group A",
                "include_path": ["survey", "survey/**"],
            },
            {
                "name": "Group B",
                "include_path": ["survey/topic-a", "survey/topic-a/**"],
            },
        ]

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self: [make_sample_paper(title="Paper 1")],
    )

    sent = []
    delays = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.batch_executor.sleep", delays.append)

    executor = BatchExecutor(config)
    executor.run()

    assert len(sent) == 2
    assert delays == [2.0]


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


@pytest.mark.parametrize("duplicate_url", [
    "http://arxiv.org/abs/2610.12345v2",
    "https://export.arxiv.org/pdf/2610.12345v3.pdf?download=1",
    "https://arxiv.org/html/2610.12345v1#section1",
])
def test_merge_arxiv_versions_keeps_scores_and_originals(duplicate_url):
    first = make_sample_paper(url="https://arxiv.org/abs/2610.12345", score=8.2)
    duplicate = make_sample_paper(url=duplicate_url, score=6.1)
    other = make_sample_paper(url="https://arxiv.org/abs/2610.54321", score=9.0)
    papers = merge_profile_results([("Robots", [first]), ("LLM", [duplicate, other])])
    assert len(papers) == 2
    assert papers[0].score == 9.0
    assert papers[1].score == 8.2
    assert papers[1].matched_profiles == {"Robots": 8.2, "LLM": 6.1}
    assert first.matched_profiles == {}
    assert duplicate.score == 6.1


def test_merge_legacy_arxiv_ids_and_missing_scores():
    first = make_sample_paper(url="https://arxiv.org/abs/hep-th/9901001v1", score=None)
    duplicate = make_sample_paper(url="https://arxiv.org/pdf/hep-th/9901001v2.pdf", score=0.0)
    unknown = make_sample_paper(url="https://arxiv.org/abs/2610.00002", score=None)
    papers = merge_profile_results([("A", [first, unknown]), ("B", [duplicate])])
    assert len(papers) == 2
    assert papers[0].score == 0.0
    assert papers[0].matched_profiles == {"A": None, "B": 0.0}
    assert papers[1].score is None


def test_merge_other_sources_by_url_not_title():
    papers = [
        make_sample_paper(source="biorxiv", url="https://example.org/1", score=4.0),
        make_sample_paper(source="biorxiv", url="https://example.org/2", score=5.0),
    ]
    result = merge_profile_results([("A", papers), ("B", [papers[0]])])
    assert len(result) == 2
    assert result[1].matched_profiles == {"A": 4.0, "B": 4.0}


@pytest.fixture
def merged_batch(config, monkeypatch):
    import smtplib
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    with open_dict(config):
        config.email.merge_profiles = True
        config.email.profile_send_delay_seconds = 60
        config.executor.max_paper_num = 1
        config.profiles = [
            {"name": name, "include_path": ["survey/**"], "max_paper_num": 1}
            for name in ["Robots", "Aircraft", "Driving", "LLM"]
        ]
    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)
    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [
        make_sample_paper(title="Shared Robot Paper", url="https://arxiv.org/abs/2610.00001"),
        make_sample_paper(title="Shared Language Paper", url="https://arxiv.org/abs/2610.00002"),
    ])
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    # Merged mode must not retain the old delay between emails.
    def unexpected_sleep(_):
        pytest.fail("Merged mode must not sleep between profiles")
    monkeypatch.setattr("zotero_arxiv_daily.batch_executor.sleep", unexpected_sleep)
    executor = BatchExecutor(config)
    return executor, sent


def test_four_profiles_send_one_deduplicated_email_with_original_scores(merged_batch, monkeypatch):
    import numpy as np

    executor, sent = merged_batch
    scores = iter([[0.9, 0.2], [0.1, 0.8], [0.7, 0.1], [0.1, 0.6]])
    calls = []
    def similarity(candidates, corpus):
        assert sent == []  # Send only after all four profiles finish.
        calls.append(len(corpus))
        return np.tile(np.array(next(scores))[:, None], (1, len(corpus)))
    monkeypatch.setattr(executor.reranker, "get_similarity_score", similarity)
    executor.run()
    assert len(calls) == 4
    assert len(sent) == 1
    raw_message = sent[0][2]
    assert "2 papers (4/4 profiles)" in _extract_subject(raw_message)
    body = email.message_from_string(raw_message).get_payload(decode=True).decode("utf-8")
    assert body.count("Shared Robot Paper") == 1
    assert body.count("Shared Language Paper") == 1
    for label in ["Robots (9.0)", "Aircraft (8.0)", "Driving (7.0)", "LLM (6.0)"]:
        assert label in body
    assert body.index("Shared Robot Paper") < body.index("Shared Language Paper")


def test_merged_partial_failure_sends_successful_results_and_reports_failure(merged_batch, monkeypatch):
    executor, sent = merged_batch
    executor.profiles[1]["include_path"] = ["missing"]
    monkeypatch.setattr(executor, "rerank_papers", lambda papers, corpus, **kw: papers[:1])
    with pytest.raises(RuntimeError, match="Aircraft"):
        executor.run()
    assert len(sent) == 1
    assert "3/4 profiles; INCOMPLETE" in _extract_subject(sent[0][2])


@pytest.mark.parametrize("send_empty", [False, True])
def test_merged_no_new_papers_respects_send_empty(merged_batch, monkeypatch, send_empty):
    executor, sent = merged_batch
    executor.config.executor.send_empty = send_empty
    monkeypatch.setattr(executor, "retrieve_all_papers", lambda: [])
    executor.run()
    assert len(sent) == int(send_empty)
    if send_empty:
        assert "0 papers (4/4 profiles)" in _extract_subject(sent[0][2])


def test_merged_all_profiles_fail_without_sending(merged_batch):
    executor, sent = merged_batch
    for profile in executor.profiles:
        profile["include_path"] = ["missing"]
    with pytest.raises(RuntimeError, match="One or more profiles failed"):
        executor.run()
    assert sent == []
