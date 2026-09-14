from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests
import re

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180
RSS_SUMMARY_PREFIX_RE = re.compile(
    r"^\s*arXiv:\S+\s+Announce Type:\s+\S+\s+Abstract:\s*",
    re.IGNORECASE,
)


def _collect_author_affiliations(authors: list[Any]) -> list[str] | None:
    seen = set()
    affiliations = []
    for author in authors:
        for affiliation in getattr(author, "affiliation", []) or []:
            normalized = str(affiliation).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                affiliations.append(normalized)
    return affiliations or None


def _arxiv_id_without_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def _rss_entry_arxiv_id(entry: Any) -> str:
    raw_id = str(entry.get("id", "")).removeprefix("oai:arXiv.org:")
    return raw_id.strip()


def _rss_entry_abs_url(entry: Any) -> str:
    link = str(entry.get("link", "")).strip()
    if link:
        return link

    for item in entry.get("links", []) or []:
        if item.get("rel") == "alternate" and item.get("href"):
            return str(item["href"]).strip()

    arxiv_id = _arxiv_id_without_version(_rss_entry_arxiv_id(entry))
    return f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""


def _rss_entry_pdf_url(entry: Any) -> str | None:
    abs_url = _rss_entry_abs_url(entry)
    if "/abs/" in abs_url:
        return abs_url.replace("/abs/", "/pdf/", 1)

    arxiv_id = _arxiv_id_without_version(_rss_entry_arxiv_id(entry))
    return f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None


def _clean_rss_summary(summary: str) -> str:
    abstract = RSS_SUMMARY_PREFIX_RE.sub("", summary).strip()
    return re.sub(r"\s+", " ", abstract)


def _split_rss_author_names(value: str) -> list[str]:
    return [author.strip() for author in value.split(",") if author.strip()]


def _rss_entry_authors(entry: Any) -> list[str]:
    raw_author_names = []
    for author in entry.get("authors", []) or []:
        name = author.get("name") if isinstance(author, dict) else getattr(author, "name", None)
        if name:
            raw_author_names.append(str(name).strip())

    for key in ("author", "dc_creator", "creator"):
        value = str(entry.get(key) or "").strip()
        if value:
            raw_author_names.append(value)

    authors = []
    seen = set()
    for raw_name in raw_author_names:
        for author in _split_rss_author_names(raw_name):
            if author not in seen:
                seen.add(author)
                authors.append(author)
    return authors


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[Any]:
        client = arxiv.Client(num_retries=0, delay_seconds=10)
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        rss_entries = [
            i for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            rss_entries = rss_entries[:10]

        all_paper_ids = [_rss_entry_arxiv_id(i) for i in rss_entries]
        if not all_paper_ids:
            return []

        # Get full information of each paper from arxiv api
        bar = None
        try:
            raw_papers = []
            bar = tqdm(total=len(all_paper_ids))
            max_batch_retries = 5
            batch_retry_delay = 30
            for i in range(0, len(all_paper_ids), 20):
                search = arxiv.Search(id_list=all_paper_ids[i:i + 20])
                for attempt in range(max_batch_retries):
                    try:
                        batch = list(client.results(search))
                        bar.update(len(batch))
                        raw_papers.extend(batch)
                        break
                    except arxiv.HTTPError as exc:
                        if exc.status == 429 and attempt < max_batch_retries - 1:
                            wait = batch_retry_delay * (attempt + 1)
                            logger.warning(f"arXiv API 429 on batch {i // 20}, retry {attempt + 1}/{max_batch_retries} in {wait}s")
                            sleep(wait)
                        else:
                            raise
                if i + 20 < len(all_paper_ids):
                    sleep(3)
            return raw_papers
        except Exception as exc:
            logger.warning(
                "Failed to retrieve full arXiv API metadata; falling back to RSS metadata only. "
                f"Reason: {type(exc).__name__}: {exc}"
            )
            return rss_entries
        finally:
            if bar is not None:
                bar.close()

    def _convert_arxiv_result_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            affiliations=_collect_author_affiliations(raw_paper.authors),
        )

    def _convert_rss_entry_to_paper(self, raw_paper: Any) -> Paper:
        return Paper(
            source=self.name,
            title=str(raw_paper.get("title", "")).strip(),
            authors=_rss_entry_authors(raw_paper),
            abstract=_clean_rss_summary(str(raw_paper.get("summary", ""))),
            url=_rss_entry_abs_url(raw_paper),
            pdf_url=_rss_entry_pdf_url(raw_paper),
            affiliations=None,
        )

    def convert_to_paper(self, raw_paper: Any) -> Paper:
        if hasattr(raw_paper, "entry_id"):
            return self._convert_arxiv_result_to_paper(raw_paper)
        return self._convert_rss_entry_to_paper(raw_paper)


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
