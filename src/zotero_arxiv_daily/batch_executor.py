from datetime import datetime

from loguru import logger
from omegaconf import DictConfig

from .executor import Executor, normalize_path_patterns


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
        for profile in self.profiles:
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

                self.send_recommendation_email(
                    reranked_papers,
                    subject=self._build_subject(profile_name),
                )
            except Exception as exc:
                logger.exception(f'Profile "{profile_name}" failed')
                failures.append((profile_name, str(exc)))

        if failures:
            failure_summary = "; ".join(f"{name}: {reason}" for name, reason in failures)
            raise RuntimeError(f"One or more profiles failed: {failure_summary}")
