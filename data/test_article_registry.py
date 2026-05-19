"""Tests for data/article_registry.py — SHA computation and ID resolution."""

from __future__ import annotations

import pytest

from data.article_registry import ArticleIdentity, ArticleRegistry
from utils.hashing import sha256_string


class TestArticleIdentity:
    """Test the ArticleIdentity dataclass."""

    def test_frozen(self) -> None:
        identity = ArticleIdentity(
            dataset="wikitext-103",
            article_id=0,
            article_sha="abc123",
            char_count=100,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            identity.article_id = 1  # type: ignore[misc]


class TestArticleRegistry:
    """Test registration, resolution, and SHA stability."""

    @pytest.fixture
    def registry(self) -> ArticleRegistry:
        reg = ArticleRegistry()
        reg.register_article("wikitext-103", 0, "Hello, world!")
        reg.register_article("wikitext-103", 1, "Another article.")
        reg.register_article("pg19", 0, "A book from Project Gutenberg.")
        return reg

    def test_register_and_resolve(self, registry: ArticleRegistry) -> None:
        identity = registry.resolve("wikitext-103", 0)
        assert identity.dataset == "wikitext-103"
        assert identity.article_id == 0
        assert identity.char_count == len("Hello, world!")

    def test_sha_matches_hashing_module(self, registry: ArticleRegistry) -> None:
        identity = registry.resolve("wikitext-103", 0)
        expected_sha = sha256_string("Hello, world!")
        assert identity.article_sha == expected_sha

    def test_sha_is_16_chars(self, registry: ArticleRegistry) -> None:
        identity = registry.resolve("wikitext-103", 0)
        assert len(identity.article_sha) == 16

    def test_different_texts_different_shas(self, registry: ArticleRegistry) -> None:
        id0 = registry.resolve("wikitext-103", 0)
        id1 = registry.resolve("wikitext-103", 1)
        assert id0.article_sha != id1.article_sha

    def test_same_text_same_sha(self) -> None:
        """Registering the same text twice yields the same SHA."""
        reg = ArticleRegistry()
        id_a = reg.register_article("test", 0, "Determinism test")
        id_b = reg.register_article("test2", 0, "Determinism test")
        assert id_a.article_sha == id_b.article_sha

    def test_resolve_unknown_raises(self, registry: ArticleRegistry) -> None:
        with pytest.raises(KeyError, match="not registered"):
            registry.resolve("wikitext-103", 999)

    def test_resolve_by_sha(self, registry: ArticleRegistry) -> None:
        expected_sha = sha256_string("Hello, world!")
        found = registry.resolve_by_sha(expected_sha)
        assert found is not None
        assert found.article_id == 0

    def test_resolve_by_sha_not_found(self, registry: ArticleRegistry) -> None:
        result = registry.resolve_by_sha("nonexistent_sha_00")
        assert result is None

    def test_len(self, registry: ArticleRegistry) -> None:
        assert len(registry) == 3

    def test_contains(self, registry: ArticleRegistry) -> None:
        assert ("wikitext-103", 0) in registry
        assert ("wikitext-103", 999) not in registry

    def test_register_corpus(self) -> None:
        """Test register_corpus with a mock CorpusLoader."""

        class MockLoader:
            dataset = "mock"

            def load(self):
                return ["Article A", "Article B", "Article C"]

        reg = ArticleRegistry()
        count = reg.register_corpus(MockLoader())  # type: ignore[arg-type]
        assert count == 3
        assert len(reg) == 3
        assert reg.resolve("mock", 0).article_sha == sha256_string("Article A")
