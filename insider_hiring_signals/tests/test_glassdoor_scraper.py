"""
Tests for Glassdoor Sentiment Scraper with AI Analysis
"""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.glassdoor_scraper import (
    GlassdoorScraper,
    GlassdoorReview,
    SentimentAnalysis,
    CompanySentimentSummary
)


class TestGlassdoorReview:
    """Tests for GlassdoorReview dataclass."""

    def test_review_creation(self):
        """Test creating a GlassdoorReview object."""
        review = GlassdoorReview(
            company_ticker='MRNA',
            company_name='Moderna',
            review_date=date(2024, 1, 15),
            overall_rating=4.0,
            ceo_approval=True,
            recommend_to_friend=True,
            business_outlook='Positive',
            pros='Great work-life balance, exciting science',
            cons='Fast-paced environment, some uncertainty',
            review_text='Overall great place to work',
            job_title='Research Scientist',
            employment_status='Current',
            source_url='https://glassdoor.com/example',
            review_id='abc123'
        )

        assert review.company_ticker == 'MRNA'
        assert review.overall_rating == 4.0
        assert 'science' in review.pros


class TestSentimentAnalysis:
    """Tests for SentimentAnalysis dataclass."""

    def test_sentiment_analysis_creation(self):
        """Test creating a SentimentAnalysis object."""
        analysis = SentimentAnalysis(
            sentiment_score=0.7,
            confidence=0.85,
            key_themes=['pipeline', 'management', 'culture'],
            mentions_layoffs=False,
            mentions_pipeline=True,
            mentions_management=True,
            mentions_culture=True,
            bullish_factors=['Strong pipeline progress', 'Transparent leadership'],
            bearish_factors=[],
            signal_weight=4,
            summary='Positive sentiment with pipeline confidence'
        )

        assert analysis.sentiment_score == 0.7
        assert analysis.mentions_pipeline is True
        assert analysis.signal_weight == 4


class TestCompanySentimentSummary:
    """Tests for CompanySentimentSummary dataclass."""

    def test_summary_creation(self):
        """Test creating a CompanySentimentSummary object."""
        summary = CompanySentimentSummary(
            company_ticker='MRNA',
            analysis_date=date.today(),
            review_count=25,
            avg_rating=3.8,
            avg_sentiment=0.3,
            sentiment_trend='improving',
            ceo_approval_rate=0.75,
            recommend_rate=0.70,
            key_concerns=['work-life balance'],
            key_positives=['innovative science', 'good pay'],
            layoff_mentions=2,
            pipeline_mentions=15,
            overall_signal='bullish',
            signal_weight=12
        )

        assert summary.company_ticker == 'MRNA'
        assert summary.sentiment_trend == 'improving'
        assert summary.overall_signal == 'bullish'


class TestGlassdoorScraper:
    """Tests for GlassdoorScraper class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.scraping = {'timeout_seconds': 30}
        config.anthropic_api_key = None
        return config

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = Mock()
        db.log_scraper_run.return_value = 1
        db.insert.return_value = 1
        return db

    @patch('scrapers.glassdoor_scraper.get_config')
    @patch('scrapers.glassdoor_scraper.get_database')
    def test_scraper_initialization(self, mock_get_db, mock_get_config, mock_config, mock_db):
        """Test scraper initialization."""
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = mock_db

        scraper = GlassdoorScraper()

        assert scraper.min_delay == 3.0
        assert scraper.max_delay == 8.0


class TestCompanyUrlConfiguration:
    """Tests for company URL configuration."""

    def test_glassdoor_ids_configured(self):
        """Test that Glassdoor IDs are configured for key companies."""
        expected_companies = ['MRNA', 'VRTX', 'BIIB', 'CRSP', 'BEAM']

        for ticker in expected_companies:
            assert ticker in GlassdoorScraper.COMPANY_GLASSDOOR_IDS

    def test_glassdoor_ids_have_required_fields(self):
        """Test that Glassdoor ID entries have required fields."""
        for ticker, info in GlassdoorScraper.COMPANY_GLASSDOOR_IDS.items():
            assert 'id' in info, f"Missing 'id' for {ticker}"
            assert 'name' in info, f"Missing 'name' for {ticker}"
            assert 'slug' in info, f"Missing 'slug' for {ticker}"


class TestSignalWeights:
    """Tests for signal weight definitions."""

    def test_signal_weights_defined(self):
        """Test that all expected signal weights are defined."""
        expected_weights = [
            'VERY_POSITIVE', 'POSITIVE', 'NEUTRAL', 'NEGATIVE', 'VERY_NEGATIVE',
            'LAYOFF_MENTIONS', 'PIPELINE_POSITIVE', 'PIPELINE_NEGATIVE'
        ]

        for weight in expected_weights:
            assert weight in GlassdoorScraper.SIGNAL_WEIGHTS

    def test_positive_weights_are_positive(self):
        """Test that positive signals have positive weights."""
        assert GlassdoorScraper.SIGNAL_WEIGHTS['VERY_POSITIVE'] > 0
        assert GlassdoorScraper.SIGNAL_WEIGHTS['POSITIVE'] > 0
        assert GlassdoorScraper.SIGNAL_WEIGHTS['PIPELINE_POSITIVE'] > 0

    def test_negative_weights_are_negative(self):
        """Test that negative signals have negative weights."""
        assert GlassdoorScraper.SIGNAL_WEIGHTS['VERY_NEGATIVE'] < 0
        assert GlassdoorScraper.SIGNAL_WEIGHTS['NEGATIVE'] < 0
        assert GlassdoorScraper.SIGNAL_WEIGHTS['LAYOFF_MENTIONS'] < 0
        assert GlassdoorScraper.SIGNAL_WEIGHTS['PIPELINE_NEGATIVE'] < 0


class TestKeywordDetection:
    """Tests for keyword detection."""

    def test_layoff_keywords_defined(self):
        """Test layoff keywords are defined."""
        expected = ['layoff', 'restructuring', 'downsizing', 'fired']
        for keyword in expected:
            assert keyword in GlassdoorScraper.LAYOFF_KEYWORDS

    def test_pipeline_keywords_defined(self):
        """Test pipeline keywords are defined."""
        expected = ['pipeline', 'clinical trial', 'fda', 'phase']
        for keyword in expected:
            assert keyword in GlassdoorScraper.PIPELINE_KEYWORDS

    def test_management_keywords_defined(self):
        """Test management keywords are defined."""
        expected = ['leadership', 'management', 'ceo', 'direction']
        for keyword in expected:
            assert keyword in GlassdoorScraper.MANAGEMENT_KEYWORDS


class TestBasicSentimentAnalysis:
    """Tests for basic (non-AI) sentiment analysis."""

    @patch('scrapers.glassdoor_scraper.get_config')
    @patch('scrapers.glassdoor_scraper.get_database')
    def test_high_rating_positive_sentiment(self, mock_get_db, mock_get_config):
        """Test that high ratings produce positive sentiment."""
        mock_config = Mock()
        mock_config.scraping = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = GlassdoorScraper()

        review = GlassdoorReview(
            company_ticker='MRNA',
            company_name='Moderna',
            review_date=date.today(),
            overall_rating=5.0,
            ceo_approval=True,
            recommend_to_friend=True,
            business_outlook='Positive',
            pros='Great company',
            cons='None',
            review_text='Love working here',
            job_title='Scientist',
            employment_status='Current',
            source_url='https://example.com'
        )

        analysis = scraper._basic_sentiment_analysis(review)

        assert analysis.sentiment_score > 0
        assert analysis.signal_weight > 0

    @patch('scrapers.glassdoor_scraper.get_config')
    @patch('scrapers.glassdoor_scraper.get_database')
    def test_low_rating_negative_sentiment(self, mock_get_db, mock_get_config):
        """Test that low ratings produce negative sentiment."""
        mock_config = Mock()
        mock_config.scraping = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = GlassdoorScraper()

        review = GlassdoorReview(
            company_ticker='MRNA',
            company_name='Moderna',
            review_date=date.today(),
            overall_rating=1.0,
            ceo_approval=False,
            recommend_to_friend=False,
            business_outlook='Negative',
            pros='None',
            cons='Everything is terrible',
            review_text='Avoid this company',
            job_title='Former Employee',
            employment_status='Former',
            source_url='https://example.com'
        )

        analysis = scraper._basic_sentiment_analysis(review)

        assert analysis.sentiment_score < 0
        assert analysis.signal_weight < 0

    @patch('scrapers.glassdoor_scraper.get_config')
    @patch('scrapers.glassdoor_scraper.get_database')
    def test_layoff_mention_detected(self, mock_get_db, mock_get_config):
        """Test that layoff mentions are detected."""
        mock_config = Mock()
        mock_config.scraping = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = GlassdoorScraper()

        review = GlassdoorReview(
            company_ticker='MRNA',
            company_name='Moderna',
            review_date=date.today(),
            overall_rating=3.0,
            ceo_approval=None,
            recommend_to_friend=None,
            business_outlook=None,
            pros='Good science',
            cons='Recent layoffs have hurt morale',
            review_text='Many people laid off recently',
            job_title='Scientist',
            employment_status='Current',
            source_url='https://example.com'
        )

        analysis = scraper._basic_sentiment_analysis(review)

        assert analysis.mentions_layoffs is True
        assert analysis.signal_weight < 0  # Layoffs should be negative

    @patch('scrapers.glassdoor_scraper.get_config')
    @patch('scrapers.glassdoor_scraper.get_database')
    def test_pipeline_mention_detected(self, mock_get_db, mock_get_config):
        """Test that pipeline mentions are detected."""
        mock_config = Mock()
        mock_config.scraping = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = GlassdoorScraper()

        review = GlassdoorReview(
            company_ticker='MRNA',
            company_name='Moderna',
            review_date=date.today(),
            overall_rating=4.0,
            ceo_approval=True,
            recommend_to_friend=True,
            business_outlook='Positive',
            pros='Exciting pipeline and strong clinical trial results',
            cons='Fast pace',
            review_text='FDA approval looking likely',
            job_title='Clinical Scientist',
            employment_status='Current',
            source_url='https://example.com'
        )

        analysis = scraper._basic_sentiment_analysis(review)

        assert analysis.mentions_pipeline is True


class TestCompanySummaryGeneration:
    """Tests for company sentiment summary generation."""

    @patch('scrapers.glassdoor_scraper.get_config')
    @patch('scrapers.glassdoor_scraper.get_database')
    def test_empty_reviews_summary(self, mock_get_db, mock_get_config):
        """Test summary generation with no reviews."""
        mock_config = Mock()
        mock_config.scraping = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = GlassdoorScraper()

        summary = scraper.generate_company_summary('MRNA', [], [])

        assert summary.review_count == 0
        assert summary.avg_rating == 0.0
        assert summary.overall_signal == 'neutral'

    @patch('scrapers.glassdoor_scraper.get_config')
    @patch('scrapers.glassdoor_scraper.get_database')
    def test_positive_reviews_summary(self, mock_get_db, mock_get_config):
        """Test summary with positive reviews."""
        mock_config = Mock()
        mock_config.scraping = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = GlassdoorScraper()

        reviews = [
            GlassdoorReview(
                company_ticker='MRNA', company_name='Moderna',
                review_date=date.today(), overall_rating=4.5,
                ceo_approval=True, recommend_to_friend=True,
                business_outlook='Positive', pros='Great', cons='None',
                review_text='', job_title='', employment_status='Current',
                source_url=''
            )
            for _ in range(5)
        ]

        analyses = [
            SentimentAnalysis(
                sentiment_score=0.5, confidence=0.8,
                key_themes=[], mentions_layoffs=False,
                mentions_pipeline=False, mentions_management=False,
                mentions_culture=False, bullish_factors=['good culture'],
                bearish_factors=[], signal_weight=3,
                summary='Positive'
            )
            for _ in range(5)
        ]

        summary = scraper.generate_company_summary('MRNA', reviews, analyses)

        assert summary.review_count == 5
        assert summary.avg_rating == 4.5
        assert summary.avg_sentiment > 0
        assert summary.overall_signal == 'bullish'


class TestDateParsing:
    """Tests for date parsing functionality."""

    @patch('scrapers.glassdoor_scraper.get_config')
    @patch('scrapers.glassdoor_scraper.get_database')
    def test_parse_standard_date(self, mock_get_db, mock_get_config):
        """Test parsing standard date format."""
        mock_config = Mock()
        mock_config.scraping = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = GlassdoorScraper()

        result = scraper._parse_date('January 15, 2024')
        assert result == date(2024, 1, 15)

    @patch('scrapers.glassdoor_scraper.get_config')
    @patch('scrapers.glassdoor_scraper.get_database')
    def test_parse_relative_date_today(self, mock_get_db, mock_get_config):
        """Test parsing 'today' relative date."""
        mock_config = Mock()
        mock_config.scraping = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = GlassdoorScraper()

        result = scraper._parse_date('Today')
        assert result == date.today()

    @patch('scrapers.glassdoor_scraper.get_config')
    @patch('scrapers.glassdoor_scraper.get_database')
    def test_parse_relative_date_days_ago(self, mock_get_db, mock_get_config):
        """Test parsing 'X days ago' format."""
        mock_config = Mock()
        mock_config.scraping = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = GlassdoorScraper()

        result = scraper._parse_date('5 days ago')
        expected = date.today() - timedelta(days=5)
        assert result == expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
