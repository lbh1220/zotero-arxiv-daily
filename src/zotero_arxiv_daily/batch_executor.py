from copy import copy
from datetime import datetime
import re
from time import sleep
from urllib.parse import urlsplit

from loguru import logger
from omegaconf import DictConfig

from .executor import Executor, normalize_path_patterns
from .protocol import Paper


def _paper_key(paper: Paper) -> tuple[str, str]:
    """Treat arXiv URL variants and versions as the same paper."""
    url = urlsplit(paper.url or paper.pdf_url or "")
    if paper.source == "arxiv" and (url.hostname or "").lower() in {
        "arxiv.org", "www.arxiv.org", "export.arxiv.org",
    }:
        paper_id = re.sub(r"^/(?:abs|pdf|html)/", "", url.path).rstrip("/")
        paper_id = re.sub(r"(?:v\d+)?(?:\.pdf)?$", "", paper_id)
        return paper.source, paper_id
    return paper.source, paper.url or paper.pdf_url or " ".join(paper.title.casefold().split())


def merge_profile_results(results: list[tuple[str, list[Paper]]]) -> list[Paper]:
    """Union profile selections, keeping every match and the highest score."""
    merged = {}
    for name, papers in results:
        for paper in papers:
            key = _paper_key(paper)
            if key not in merged:
                merged[key] = copy(paper)
                merged[key].matched_profiles = {}
            entry = merged[key]
            entry.matched_profiles[name] = paper.score
            if paper.score is not None and (entry.score is None or paper.score > entry.score):
                entry.score = paper.score
    return sorted(
        merged.values(),
        key=lambda paper: paper.score if paper.score is not None else float("-inf"),
        reverse=True,
    )


class BatchExecutor(Executor):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.profiles = self._load_profiles(config)

    def _load_profiles(self, config: DictConfig) -> list[dict]:
        raw_profiles = config.get("profiles")
        if not raw_profiles:
            raise ValueError("config.profiles must be a non-empty list when using batch mode.")

        profiles = []
        for index, profile in enumerate(raw_profiles):
            config_prefix = f"profiles[{index}]"
            name = profile.get("name")
            if not isinstance(name, str) or not name.strip():
                raise TypeError(f"config.{config_prefix}.name must be a non-empty string.")

            include_path = normalize_path_patterns(
                profile.get("include_path"),
                "include_path",
                config_prefix=config_prefix,
            )
            ignore_path = normalize_path_patterns(
                profile.get("ignore_path"),
                "ignore_path",
                config_prefix=config_prefix,
            )
            if include_path is None and ignore_path is None:
                raise ValueError(
                    f"config.{config_prefix} must define include_path, ignore_path, or both."
                )

            max_paper_num = profile.get("max_paper_num")
            if max_paper_num is not None and (not isinstance(max_paper_num, int) or max_paper_num <= 0):
                raise TypeError(f"config.{config_prefix}.max_paper_num must be a positive integer or null.")

            profiles.append(
                {
                    "name": name.strip(),
                    "include_path": include_path,
                    "ignore_path": ignore_path,
                    "max_paper_num": max_paper_num,
                }
            )
        return profiles

    def _build_subject(self, profile_name: str) -> str:
        today = datetime.now().strftime("%Y/%m/%d")
        return f"Daily arXiv - {profile_name} - {today}"

    def run(self):
        corpus = self.fetch_zotero_corpus()
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return

        all_papers = self.retrieve_all_papers()
        if len(all_papers) == 0 and not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return

        failures = []
        merge_profiles = self.config.email.get("merge_profiles", True)
        profile_results = []
        send_delay = max(0.0, float(self.config.email.get("profile_send_delay_seconds", 60)))
        for index, profile in enumerate(self.profiles):
            profile_name = profile["name"]
            logger.info(f"Processing profile: {profile_name}")
            try:
                profile_corpus = self.filter_corpus_with_patterns(
                    corpus,
                    profile["include_path"],
                    profile["ignore_path"],
                )
                if len(profile_corpus) == 0:
                    raise ValueError(
                        f'No zotero papers matched profile "{profile_name}". '
                        "Please check include_path/ignore_path settings."
                    )

                reranked_papers = []
                if len(all_papers) > 0:
                    reranked_papers = self.rerank_papers(
                        all_papers,
                        profile_corpus,
                        max_paper_num=profile["max_paper_num"],
                    )
                    self.enrich_papers(reranked_papers)

                if merge_profiles:
                    # Rerankers mutate shared Paper scores on every profile.
                    profile_results.append((profile_name, [copy(p) for p in reranked_papers]))
                    continue

                self.send_recommendation_email(
                    reranked_papers,
                    subject=self._build_subject(profile_name),
                )
                if send_delay > 0 and index < len(self.profiles) - 1:
                    logger.info(f"Waiting {send_delay}s before sending the next profile email")
                    sleep(send_delay)
            except Exception as exc:
                logger.exception(f'Profile "{profile_name}" failed')
                failures.append((profile_name, str(exc)))

        if merge_profiles and profile_results:
            papers = merge_profile_results(profile_results)
            today = datetime.now().strftime("%Y/%m/%d")
            warning = "; INCOMPLETE" if failures else ""
            subject = (
                f"Daily arXiv - {len(papers)} papers "
                f"({len(profile_results)}/{len(self.profiles)} profiles{warning}) - {today}"
            )
            self.send_recommendation_email(papers, subject=subject)

        if failures:
            failure_summary = "; ".join(f"{name}: {reason}" for name, reason in failures)
            raise RuntimeError(f"One or more profiles failed: {failure_summary}")
