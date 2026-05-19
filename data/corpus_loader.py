"""Deterministic corpus loaders for wikitext-103 and PG19.

Each loader returns articles as a list of strings. Sampling is deterministic
given a seed — the same ``(dataset, article_id, seed)`` triple always
returns the same article text.
"""

from __future__ import annotations

from typing import List, Optional

from utils.logger import get_logger

log = get_logger(__name__)


class CorpusLoader:
    """Load and sample articles from wikitext-103 or PG19.

    Parameters
    ----------
    dataset : str
        ``"wikitext-103"`` or ``"pg19"``.
    cache_dir : str, optional
        HuggingFace datasets cache directory.
    """

    _SUPPORTED_DATASETS = {"wikitext-103", "pg19"}

    def __init__(self, dataset: str, cache_dir: Optional[str] = None) -> None:
        if dataset not in self._SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset!r}.  "
                f"Choose from {self._SUPPORTED_DATASETS}"
            )
        self.dataset = dataset
        self.cache_dir = cache_dir
        self._articles: Optional[List[str]] = None

    def _load_wikitext103(self) -> List[str]:
        """Load wikitext-103 and split into articles."""
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split="test",
            cache_dir=self.cache_dir,
        )
        # Concatenate all text, then split on article boundaries (double newlines
        # following a title pattern "= Title =" at the start of a line)
        full_text = "\n".join(row["text"] for row in ds)
        articles = self._split_into_articles(full_text)
        log.info("Loaded wikitext-103: %d articles", len(articles))
        return articles

    def _load_pg19(self) -> List[str]:
        """Load PG19 test split and return individual books."""
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset(
            "deepmind/pg19",
            split="test",
            cache_dir=self.cache_dir,
        )
        articles = [row["text"] for row in ds]
        log.info("Loaded pg19: %d articles", len(articles))
        return articles

    @staticmethod
    def _split_into_articles(text: str) -> List[str]:
        """Split raw wikitext into individual articles.

        Articles are delimited by lines starting with ``" = "`` (level-1
        headings in the wikitext markup).  Empty articles are dropped.
        """
        import re

        # Split on level-1 headings (single = on each side)
        parts = re.split(r"\n(?= = [^=])", text)
        articles = [p.strip() for p in parts if p.strip()]
        return articles

    def load(self) -> List[str]:
        """Load and cache the article list. Idempotent."""
        if self._articles is not None:
            return self._articles

        if self.dataset == "wikitext-103":
            self._articles = self._load_wikitext103()
        elif self.dataset == "pg19":
            self._articles = self._load_pg19()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        return self._articles

    def get_article(self, article_id: int) -> str:
        """Return the article at index *article_id*.

        Raises
        ------
        IndexError
            If *article_id* is out of range.
        """
        articles = self.load()
        if article_id < 0 or article_id >= len(articles):
            raise IndexError(
                f"article_id {article_id} out of range for "
                f"{self.dataset} ({len(articles)} articles)"
            )
        return articles[article_id]

    def num_articles(self) -> int:
        """Return the number of articles in the loaded corpus."""
        return len(self.load())

    def sample_articles(
        self, n: int, seed: int = 42
    ) -> List[str]:
        """Deterministically sample *n* articles using *seed*.

        Parameters
        ----------
        n : int
            Number of articles to sample.
        seed : int
            RNG seed for reproducible sampling.

        Returns
        -------
        list[str]
            Sampled article texts.
        """
        import random as _random

        articles = self.load()
        if n > len(articles):
            raise ValueError(
                f"Requested {n} articles but only {len(articles)} available"
            )

        rng = _random.Random(seed)
        indices = rng.sample(range(len(articles)), n)
        indices.sort()  # Deterministic ordering
        return [articles[i] for i in indices]
