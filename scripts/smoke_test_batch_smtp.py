#!/usr/bin/env python3
"""Run a local SMTP smoke test for multi-profile email delivery.

This script uses the project's real BatchExecutor and send_email flow, while
stubbing remote dependencies (Zotero, arXiv, OpenAI) and sending messages to a
local SMTP server started in-process.
"""

from __future__ import annotations

import argparse
import email
from email.header import decode_header, make_header
from pathlib import Path
import socket
import smtplib
import sys

import aiosmtpd.controller as smtp_controller
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.canned_responses import (  # noqa: E402
    make_sample_paper,
    make_stub_openai_client,
    make_stub_zotero_client,
)
from zotero_arxiv_daily.batch_executor import BatchExecutor  # noqa: E402


class MailboxHandler:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def handle_DATA(self, server, session, envelope):  # noqa: N802
        self.messages.append(envelope.content.decode("utf-8", errors="replace"))
        return "250 Message accepted for delivery"


def decode_subject(raw_message: str) -> str:
    message = email.message_from_string(raw_message)
    return str(make_header(decode_header(message["Subject"])))


def build_config(smtp_port: int):
    base_cfg = OmegaConf.load(ROOT / "config" / "base.yaml")
    my_cfg = OmegaConf.load(ROOT / "config" / "my_conf.yaml")
    cfg = OmegaConf.merge(base_cfg, my_cfg)

    cfg.zotero.user_id = "000000"
    cfg.zotero.api_key = "fake-zotero-key"
    cfg.email.sender = "test@example.com"
    cfg.email.receiver = "test@example.com"
    cfg.email.smtp_server = "127.0.0.1"
    cfg.email.smtp_port = smtp_port
    cfg.email.sender_password = "test"
    cfg.llm.api.key = "sk-fake"
    cfg.llm.api.base_url = "http://localhost:30000/v1"
    cfg.llm.generation_kwargs.model = "gpt-4o-mini"
    cfg.reranker.api.key = "sk-fake"
    cfg.reranker.api.base_url = "http://localhost:30000/v1"
    cfg.reranker.api.model = "text-embedding-3-large"
    cfg.executor.source = ["arxiv"]
    cfg.executor.reranker = "api"
    cfg.executor.debug = False
    cfg.executor.send_empty = False
    return cfg


def build_stub_zotero():
    collections = [
        {"key": "MOBILE", "data": {"name": "mobile robot", "parentCollection": False}},
        {"key": "RESEARCH", "data": {"name": "Research", "parentCollection": False}},
        {"key": "AIRCRAFT", "data": {"name": "Aircraft in city", "parentCollection": False}},
        {"key": "UAM", "data": {"name": "UAM and UAS", "parentCollection": False}},
        {"key": "AUTO", "data": {"name": "autonomous driving", "parentCollection": False}},
        {"key": "LLM", "data": {"name": "LLM", "parentCollection": False}},
    ]
    items = [
        {
            "data": {
                "title": "Mobile Robot Paper",
                "abstractNote": "Robot mapping and planning.",
                "dateAdded": "2026-01-15T10:00:00Z",
                "collections": ["MOBILE"],
            }
        },
        {
            "data": {
                "title": "Research Overview",
                "abstractNote": "General research paper.",
                "dateAdded": "2026-01-16T10:00:00Z",
                "collections": ["RESEARCH"],
            }
        },
        {
            "data": {
                "title": "Aircraft City Mobility",
                "abstractNote": "Urban aircraft systems.",
                "dateAdded": "2026-01-17T10:00:00Z",
                "collections": ["AIRCRAFT"],
            }
        },
        {
            "data": {
                "title": "UAM Systems",
                "abstractNote": "UAM vehicle study.",
                "dateAdded": "2026-01-18T10:00:00Z",
                "collections": ["UAM"],
            }
        },
        {
            "data": {
                "title": "Autonomous Driving Paper",
                "abstractNote": "Driving policy learning.",
                "dateAdded": "2026-01-19T10:00:00Z",
                "collections": ["AUTO"],
            }
        },
        {
            "data": {
                "title": "LLM Paper",
                "abstractNote": "Language model alignment.",
                "dateAdded": "2026-01-20T10:00:00Z",
                "collections": ["LLM"],
            }
        },
    ]
    return make_stub_zotero_client(collections=collections, items=items)


def install_stubs():
    import zotero_arxiv_daily.executor as executor_mod
    import zotero_arxiv_daily.reranker.api as reranker_api_mod
    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    import zotero_arxiv_daily.retriever.base as retriever_base_mod
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    stub_client = make_stub_openai_client()
    executor_mod.zotero.Zotero = lambda *args, **kwargs: build_stub_zotero()
    executor_mod.OpenAI = lambda **kwargs: stub_client
    reranker_api_mod.OpenAI = lambda **kwargs: stub_client
    retriever_base_mod.sleep = lambda _: None
    registered_retrievers["arxiv"].retrieve_papers = lambda self: [
        make_sample_paper(title="Candidate Paper 1", url="https://arxiv.org/abs/1", pdf_url="https://arxiv.org/pdf/1"),
        make_sample_paper(title="Candidate Paper 2", url="https://arxiv.org/abs/2", pdf_url="https://arxiv.org/pdf/2"),
        make_sample_paper(title="Candidate Paper 3", url="https://arxiv.org/abs/3", pdf_url="https://arxiv.org/pdf/3"),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the batch email SMTP smoke test.")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Local SMTP port to bind. Defaults to 0 for an ephemeral free port.",
    )
    parser.add_argument(
        "--inject-broken-at",
        type=int,
        default=None,
        help="Insert a failing profile at the given 0-based position to test continue-on-error behavior.",
    )
    return parser.parse_args()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def inject_broken_profile(config, index: int | None):
    if index is None:
        return config
    profiles = list(config.profiles)
    broken_profile = OmegaConf.create(
        {
            "name": "Broken",
            "include_path": ["__not_exist__", "__not_exist__/**"],
        }
    )
    insert_at = max(0, min(index, len(profiles)))
    profiles.insert(insert_at, broken_profile)
    config.profiles = profiles
    return config


def main() -> int:
    args = parse_args()
    handler = MailboxHandler()
    original_get_localhost = smtp_controller.get_localhost
    smtp_controller.get_localhost = lambda: "127.0.0.1"
    smtp_port = args.port if args.port != 0 else find_free_port()
    controller = smtp_controller.Controller(handler, hostname="127.0.0.1", port=smtp_port)

    original_login = smtplib.SMTP.login
    original_ssl = smtplib.SMTP_SSL
    smtplib.SMTP.login = lambda self, user, password: None

    class DisabledSMTPSSL:
        def __init__(self, *args, **kwargs):
            raise OSError("SSL disabled in local smoke test")

    smtplib.SMTP_SSL = DisabledSMTPSSL

    controller_started = False
    executor_error: Exception | None = None
    try:
        controller.start()
        controller_started = True
        install_stubs()
        config = build_config(smtp_port)
        inject_broken_profile(config, args.inject_broken_at)
        executor = BatchExecutor(config)
        try:
            executor.run()
        except Exception as exc:  # Keep the SMTP summary even when batch mode fails overall.
            executor_error = exc
    finally:
        smtp_controller.get_localhost = original_get_localhost
        smtplib.SMTP.login = original_login
        smtplib.SMTP_SSL = original_ssl
        if controller_started:
            controller.stop()

    subjects = [decode_subject(message) for message in handler.messages]
    expected_profiles = [profile.name for profile in config.profiles if profile.name != "Broken"]

    print(f"Captured {len(handler.messages)} messages from local SMTP.")
    for subject in subjects:
        print(f" - {subject}")

    if executor_error is not None:
        print(f"Executor raised after processing profiles: {executor_error}", file=sys.stderr)

    if len(handler.messages) != len(expected_profiles):
        print("Unexpected email count.", file=sys.stderr)
        return 1

    missing = [profile for profile in expected_profiles if not any(profile in subject for subject in subjects)]
    if missing:
        print(f"Missing subjects for profiles: {missing}", file=sys.stderr)
        return 1

    if executor_error is not None:
        return 1

    print("SMTP smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
