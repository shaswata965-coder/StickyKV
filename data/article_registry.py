"""SHA-based article identity resolution.

Provides a registry that maps ``(dataset, article_id)`` to a stable SHA
fingerprint. This allows cross-run identity verification: if two runs
reference the same ``article_sha``, they used the same text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.hashing import sha256_string
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ArticleIdentity:
    """Immutable identity record for a single article."""

    dataset: str
    article_id: int
    article_sha: str
    char_count: int


class ArticleRegistry:
    """Build and query a registry of article SHA fingerprints.

    Typical usage::

        loader = CorpusLoader("wikitext-103")
        registry = ArticleRegistry()
        registry.register_corpus(loader)

        identity = registry.resolve("wikitext-103", 0)
        print(identity.article_sha)
    """

    def __init__(self) -> None:
        self._registry: dict[tuple[str, int], ArticleIdentity] = {}

    def register_article(
        self, dataset: str, article_id: int, text: str
    ) -> ArticleIdentity:
        """Compute and store the SHA for a single article.

        Parameters
        ----------
        dataset : str
            Dataset name (e.g. ``"wikitext-103"``).
        article_id : int
            Zero-based article index.
        text : str
            Full article text.

        Returns
        -------
        ArticleIdentity
        """
        sha = sha256_string(text)
        identity = ArticleIdentity(
            dataset=dataset,
            article_id=article_id,
            article_sha=sha,
            char_count=len(text),
        )
        self._registry[(dataset, article_id)] = identity
        return identity

    def register_corpus(self, loader: "CorpusLoader") -> int:  # type: ignore[name-defined]
        """Register all articles from a ``CorpusLoader``.

        Parameters
        ----------
        loader : data.corpus_loader.CorpusLoader
            A loader with its corpus already loaded (or it will be loaded
            on first access).

        Returns
        -------
        int
            Number of articles registered.
        """
        articles = loader.load()
        for idx, text in enumerate(articles):
            self.register_article(loader.dataset, idx, text)
        log.info(
            "Registered %d articles from %s", len(articles), loader.dataset
        )
        return len(articles)

    def resolve(self, dataset: str, article_id: int) -> ArticleIdentity:
        """Look up the identity for ``(dataset, article_id)``.

        Raises
        ------
        KeyError
            If the article has not been registered.
        """
        key = (dataset, article_id)
        if key not in self._registry:
            raise KeyError(
                f"Article ({dataset!r}, {article_id}) not registered.  "
                f"Call register_article or register_corpus first."
            )
        return self._registry[key]

    def resolve_by_sha(
        self, article_sha: str
    ) -> Optional[ArticleIdentity]:
        """Find an article by its SHA fingerprint.

        Returns
        -------
        ArticleIdentity or None
        """
        for identity in self._registry.values():
            if identity.article_sha == article_sha:
                return identity
        return None

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, key: tuple[str, int]) -> bool:
        return key in self._registry
