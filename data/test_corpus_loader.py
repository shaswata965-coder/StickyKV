"""Tests for data/corpus_loader.py — deterministic sampling, no GPU required."""

from __future__ import annotations

import pytest

from data.corpus_loader import CorpusLoader


class TestCorpusLoaderValidation:
    """Test input validation without loading data."""

    def test_rejects_unknown_dataset(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dataset"):
            CorpusLoader("imagenet")

    def test_accepts_wikitext(self) -> None:
        loader = CorpusLoader("wikitext-103")
        assert loader.dataset == "wikitext-103"

    def test_accepts_pg19(self) -> None:
        loader = CorpusLoader("pg19")
        assert loader.dataset == "pg19"


class TestArticleSplitting:
    """Test the wikitext article splitting logic on synthetic data."""

    def test_split_into_articles(self) -> None:
        text = (
            " = Article One =\nSome content here.\n\n"
            " = Article Two =\nMore content.\n\n"
            " = Article Three =\nFinal piece."
        )
        articles = CorpusLoader._split_into_articles(text)
        assert len(articles) >= 2  # At least two articles from the splits

    def test_empty_articles_dropped(self) -> None:
        text = " = Title =\n\n\n\n = Another =\nContent"
        articles = CorpusLoader._split_into_articles(text)
        for article in articles:
            assert article.strip() != ""


class TestDeterministicSampling:
    """Test that sampling is deterministic given a fixed seed.

    Uses a mock corpus (monkey-patched) to avoid dataset downloads.
    """

    @pytest.fixture
    def mock_loader(self) -> CorpusLoader:
        """Create a loader with pre-loaded mock articles."""
        loader = CorpusLoader("wikitext-103")
        loader._articles = [f"Article {i}: content {i * 7}" for i in range(100)]
        return loader

    def test_same_seed_same_result(self, mock_loader: CorpusLoader) -> None:
        sample_a = mock_loader.sample_articles(10, seed=42)
        sample_b = mock_loader.sample_articles(10, seed=42)
        assert sample_a == sample_b

    def test_different_seed_different_result(self, mock_loader: CorpusLoader) -> None:
        sample_a = mock_loader.sample_articles(10, seed=42)
        sample_b = mock_loader.sample_articles(10, seed=99)
        assert sample_a != sample_b

    def test_sample_preserves_order(self, mock_loader: CorpusLoader) -> None:
        """Samples should be in sorted (ascending) index order."""
        sample = mock_loader.sample_articles(10, seed=42)
        # Each article starts with "Article N:" — extract N
        indices = [int(s.split(":")[0].split(" ")[1]) for s in sample]
        assert indices == sorted(indices)

    def test_sample_too_many_raises(self, mock_loader: CorpusLoader) -> None:
        with pytest.raises(ValueError, match="Requested 200"):
            mock_loader.sample_articles(200, seed=42)


class TestGetArticle:
    """Test article retrieval by index."""

    @pytest.fixture
    def mock_loader(self) -> CorpusLoader:
        loader = CorpusLoader("wikitext-103")
        loader._articles = ["zero", "one", "two"]
        return loader

    def test_get_valid_article(self, mock_loader: CorpusLoader) -> None:
        assert mock_loader.get_article(0) == "zero"
        assert mock_loader.get_article(2) == "two"

    def test_get_out_of_range(self, mock_loader: CorpusLoader) -> None:
        with pytest.raises(IndexError):
            mock_loader.get_article(999)

    def test_get_negative_index(self, mock_loader: CorpusLoader) -> None:
        with pytest.raises(IndexError):
            mock_loader.get_article(-1)

    def test_num_articles(self, mock_loader: CorpusLoader) -> None:
        assert mock_loader.num_articles() == 3
