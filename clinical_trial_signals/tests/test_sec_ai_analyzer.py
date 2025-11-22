"""
Tests for the AI-Powered SEC Filing Analysis module.
"""
import pytest
from datetime import date
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.sec_ai_analyzer import (
    SECAIAnalyzer,
    MockSECAIAnalyzer,
    AIAnalysisResult,
    SentimentCategory,
    LanguagePattern,
    get_analyzer,
)


class TestMockSECAIAnalyzer:
    """Tests for the mock analyzer (no API calls)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MockSECAIAnalyzer()

        self.positive_filing = """
        FORM 8-K

        Item 8.01 Other Events

        BioTech Corp announced positive topline results from its Phase 3 trial.
        The trial met its primary endpoint with statistically significant results.
        Management is confident in the path forward and expects FDA approval.
        This breakthrough therapy shows strong efficacy and favorable safety.
        """

        self.negative_filing = """
        FORM 8-K

        Item 8.01 Other Events

        BioTech Corp announced that its Phase 3 trial failed to meet the primary
        endpoint. The trial was terminated due to safety concerns and adverse events.
        Development has been discontinued. There is no assurance of future success.
        """

        self.neutral_filing = """
        FORM 8-K

        Item 7.01 Regulation FD Disclosure

        BioTech Corp provided a routine business update.
        Operations continue as planned.
        """

        self.metadata = {
            "company_name": "BioTech Corp",
            "ticker": "BTCH",
            "filing_type": "8-K",
            "filing_date": "2024-01-15",
            "accession_number": "0001234567-24-000001"
        }

    def test_analyze_positive_filing(self):
        """Test analysis of a positive filing."""
        result = self.analyzer.analyze_filing(
            self.positive_filing,
            self.metadata,
            filing_id="test_positive"
        )

        assert result is not None
        assert result.sentiment_score > 0
        assert result.sentiment_category in (SentimentCategory.BULLISH, SentimentCategory.VERY_BULLISH)
        assert result.filing_id == "test_positive"
        assert result.model_used == "mock-model"

    def test_analyze_negative_filing(self):
        """Test analysis of a negative filing."""
        result = self.analyzer.analyze_filing(
            self.negative_filing,
            self.metadata,
            filing_id="test_negative"
        )

        assert result is not None
        assert result.sentiment_score < 0
        assert result.sentiment_category in (SentimentCategory.BEARISH, SentimentCategory.VERY_BEARISH)

    def test_analyze_neutral_filing(self):
        """Test analysis of a neutral filing."""
        result = self.analyzer.analyze_filing(
            self.neutral_filing,
            self.metadata,
            filing_id="test_neutral"
        )

        assert result is not None
        assert result.sentiment_category == SentimentCategory.NEUTRAL

    def test_score_to_signal_weight_positive(self):
        """Test score to signal weight conversion for positive scores."""
        assert self.analyzer.score_to_signal_weight(9.0) == 5
        assert self.analyzer.score_to_signal_weight(6.0) == 4
        assert self.analyzer.score_to_signal_weight(3.0) == 2
        assert self.analyzer.score_to_signal_weight(0.0) == 0

    def test_score_to_signal_weight_negative(self):
        """Test score to signal weight conversion for negative scores."""
        assert self.analyzer.score_to_signal_weight(-3.0) == -2
        assert self.analyzer.score_to_signal_weight(-6.0) == -4
        assert self.analyzer.score_to_signal_weight(-9.0) == -5

    def test_analyze_and_create_signal(self):
        """Test creating a signal from analysis."""
        signal = self.analyzer.analyze_and_create_signal(
            self.positive_filing,
            self.metadata,
            trial_id="NCT12345678"
        )

        assert signal is not None
        assert signal["trial_id"] == "NCT12345678"
        assert signal["signal_type"] == "sec_ai_positive"
        assert signal["signal_weight"] > 0
        assert "raw_data" in signal
        assert signal["raw_data"]["sentiment_score"] > 0

    def test_analyze_and_create_negative_signal(self):
        """Test creating a negative signal from analysis."""
        signal = self.analyzer.analyze_and_create_signal(
            self.negative_filing,
            self.metadata,
            trial_id="NCT87654321"
        )

        assert signal is not None
        assert signal["signal_type"] == "sec_ai_negative"
        assert signal["signal_weight"] < 0

    def test_hedging_language_detection(self):
        """Test detection of hedging language."""
        hedging_content = """
        This may result in positive outcomes, though there might be risks.
        We could potentially achieve our goals, but no assurance can be given.
        Results are uncertain and subject to various factors.
        """

        result = self.analyzer.analyze_filing(hedging_content, self.metadata)

        assert result is not None
        assert result.hedging_score > 0.3  # Should detect hedging

    def test_confident_language_detection(self):
        """Test detection of confident language."""
        confident_content = """
        We will achieve our primary endpoint. We expect strong results.
        Management is confident in the data. We are on track for FDA submission.
        We remain committed to our timelines.
        """

        result = self.analyzer.analyze_filing(confident_content, self.metadata)

        assert result is not None
        assert result.confidence_language_score > 0.3  # Should detect confidence


class TestSECAIAnalyzer:
    """Tests for the real analyzer (mocking API calls)."""

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': ''}):
            analyzer = SECAIAnalyzer(api_key=None)
            assert analyzer.api_key == ""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        analyzer = SECAIAnalyzer(api_key="test_key")
        assert analyzer.api_key == "test_key"
        assert analyzer.model == SECAIAnalyzer.DEFAULT_MODEL

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        analyzer = SECAIAnalyzer(api_key="test_key", model="custom-model")
        assert analyzer.model == "custom-model"

    @patch('scrapers.sec_ai_analyzer.SECAIAnalyzer.client')
    def test_analyze_filing_api_call(self, mock_client):
        """Test that analyze_filing makes correct API call."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='''
        {
            "sentiment_score": 7.5,
            "sentiment_category": "bullish",
            "confidence": 0.8,
            "hedging_score": 0.2,
            "confidence_language_score": 0.7,
            "key_insights": ["Positive trial results"],
            "trial_mentions": [],
            "forward_looking_statements": [],
            "risk_factors": [],
            "language_patterns": [],
            "summary": "Positive filing"
        }
        ''')]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=200)

        mock_client.messages.create.return_value = mock_response

        analyzer = SECAIAnalyzer(api_key="test_key")

        # Need to provide enough content to pass the length check (>100 chars)
        long_content = "Test filing content with positive results. " * 10

        result = analyzer.analyze_filing(
            long_content,
            {"company_name": "Test Co", "ticker": "TEST"}
        )

        assert result is not None
        assert result.sentiment_score == 7.5
        assert result.sentiment_category == SentimentCategory.BULLISH

    def test_analyze_filing_no_api_key(self):
        """Test that analyze_filing returns None without API key."""
        analyzer = SECAIAnalyzer(api_key="")
        result = analyzer.analyze_filing(
            "Test content",
            {"company_name": "Test"}
        )
        assert result is None

    def test_analyze_filing_empty_content(self):
        """Test that analyze_filing handles empty content."""
        analyzer = SECAIAnalyzer(api_key="test_key")
        result = analyzer.analyze_filing("short", {"company_name": "Test"})
        assert result is None


class TestGetAnalyzer:
    """Tests for the analyzer factory function."""

    def test_get_mock_analyzer(self):
        """Test getting mock analyzer."""
        analyzer = get_analyzer(use_mock=True)
        assert isinstance(analyzer, MockSECAIAnalyzer)

    def test_get_mock_analyzer_without_api_key(self):
        """Test getting mock analyzer when no API key."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': ''}):
            analyzer = get_analyzer(use_mock=False)
            assert isinstance(analyzer, MockSECAIAnalyzer)


class TestAIAnalysisResult:
    """Tests for AIAnalysisResult dataclass."""

    def test_result_creation(self):
        """Test creating an AIAnalysisResult."""
        result = AIAnalysisResult(
            filing_id="test_123",
            sentiment_score=7.5,
            sentiment_category=SentimentCategory.BULLISH,
            confidence=0.85,
            hedging_score=0.2,
            confidence_language_score=0.7,
            key_insights=["Insight 1", "Insight 2"],
            trial_mentions=[],
            forward_looking_statements=[],
            risk_factors=[],
            language_patterns=[],
            raw_analysis={},
            processing_time_ms=150,
            model_used="test-model",
            tokens_used=300
        )

        assert result.filing_id == "test_123"
        assert result.sentiment_score == 7.5
        assert len(result.key_insights) == 2


class TestLanguagePattern:
    """Tests for LanguagePattern dataclass."""

    def test_pattern_creation(self):
        """Test creating a LanguagePattern."""
        pattern = LanguagePattern(
            pattern_type="hedging",
            text_excerpt="may result in",
            confidence=0.9
        )

        assert pattern.pattern_type == "hedging"
        assert pattern.confidence == 0.9
