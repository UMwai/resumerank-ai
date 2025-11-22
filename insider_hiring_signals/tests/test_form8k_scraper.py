"""
Tests for Form 8-K Executive Changes Scraper
"""

import pytest
from datetime import date, datetime
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.form8k_scraper import (
    Form8KScraper,
    ExecutiveChange,
    Form8KFiling
)


class TestExecutiveChange:
    """Tests for ExecutiveChange dataclass."""

    def test_executive_change_creation(self):
        """Test creating an ExecutiveChange object."""
        change = ExecutiveChange(
            company_ticker='MRNA',
            company_cik='0001682852',
            executive_name='John Smith',
            title='CEO',
            change_type='Departure',
            effective_date=date(2024, 1, 15),
            announcement_date=date(2024, 1, 10),
            reason='retirement',
            is_voluntary=True,
            successor_name='Jane Doe',
            filing_url='https://www.sec.gov/example',
            filing_text='Item 5.02 disclosure text',
            severity_score=6,
            signal_weight=-6
        )

        assert change.company_ticker == 'MRNA'
        assert change.executive_name == 'John Smith'
        assert change.change_type == 'Departure'
        assert change.is_voluntary is True


class TestForm8KFiling:
    """Tests for Form8KFiling dataclass."""

    def test_form8k_filing_creation(self):
        """Test creating a Form8KFiling object."""
        filing = Form8KFiling(
            accession_number='0001234-24-000001',
            company_name='Test Company Inc',
            company_cik='0001234567',
            filing_date=date(2024, 1, 15),
            filing_url='https://www.sec.gov/example',
            items=['5.02', '7.01'],
            has_item_502=True
        )

        assert filing.accession_number == '0001234-24-000001'
        assert filing.has_item_502 is True
        assert '5.02' in filing.items


class TestForm8KScraper:
    """Tests for Form8KScraper class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.sec_user_agent = 'Test Agent test@example.com'
        config.scraping = {'timeout_seconds': 30}
        config.watchlist = ['MRNA', 'VRTX', 'BIIB']
        config.sec_edgar = {'rate_limit_requests_per_second': 10}
        config.anthropic_api_key = None
        return config

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = Mock()
        db.log_scraper_run.return_value = 1
        return db

    @patch('scrapers.form8k_scraper.get_config')
    @patch('scrapers.form8k_scraper.get_database')
    def test_scraper_initialization(self, mock_get_db, mock_get_config, mock_config, mock_db):
        """Test scraper initialization."""
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = mock_db

        scraper = Form8KScraper(use_ai=False)

        assert scraper.use_ai is False
        assert scraper.rate_limit == 10


class TestItem502Detection:
    """Tests for Item 5.02 detection patterns."""

    def test_item_502_patterns_defined(self):
        """Test that Item 5.02 patterns are defined."""
        assert len(Form8KScraper.ITEM_502_PATTERNS) > 0

    def test_item_502_pattern_matches(self):
        """Test Item 5.02 pattern matching."""
        import re

        test_texts = [
            'Item 5.02 - Departure of Directors',
            'ITEM 5.02 DEPARTURE OF DIRECTORS',
            'Item 5.02: Election of Directors',
            'Departure of Directors or Certain Officers',
            'Resignation of Chief Executive Officer'
        ]

        for text in test_texts:
            matched = any(
                re.search(pattern, text.lower())
                for pattern in Form8KScraper.ITEM_502_PATTERNS
            )
            assert matched, f"Pattern should match: {text}"


class TestSignalWeights:
    """Tests for signal weight definitions."""

    def test_signal_weights_defined(self):
        """Test that all expected signal weights are defined."""
        expected_weights = [
            'CEO_DEPARTURE', 'CFO_DEPARTURE', 'CMO_DEPARTURE',
            'CEO_APPOINTMENT', 'CMO_APPOINTMENT',
            'INTERIM_APPOINTMENT', 'MASS_EXODUS'
        ]

        for weight in expected_weights:
            assert weight in Form8KScraper.SIGNAL_WEIGHTS

    def test_departures_are_negative(self):
        """Test that departure signals are negative."""
        departure_weights = [
            'CEO_DEPARTURE', 'CFO_DEPARTURE', 'CMO_DEPARTURE',
            'CSO_DEPARTURE', 'COO_DEPARTURE'
        ]

        for weight in departure_weights:
            assert Form8KScraper.SIGNAL_WEIGHTS[weight] < 0

    def test_appointments_are_positive(self):
        """Test that appointment signals are positive."""
        appointment_weights = [
            'CEO_APPOINTMENT', 'CMO_APPOINTMENT', 'CFO_APPOINTMENT'
        ]

        for weight in appointment_weights:
            assert Form8KScraper.SIGNAL_WEIGHTS[weight] > 0

    def test_cmo_departure_high_weight(self):
        """Test CMO departure has high negative weight (critical for biotech)."""
        assert Form8KScraper.SIGNAL_WEIGHTS['CMO_DEPARTURE'] <= -5


class TestExecutiveTitleNormalization:
    """Tests for executive title normalization."""

    @patch('scrapers.form8k_scraper.get_config')
    @patch('scrapers.form8k_scraper.get_database')
    def test_normalize_ceo_titles(self, mock_get_db, mock_get_config):
        """Test CEO title normalization."""
        mock_config = Mock()
        mock_config.sec_user_agent = 'Test'
        mock_config.scraping = {}
        mock_config.sec_edgar = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = Form8KScraper(use_ai=False)

        assert scraper._normalize_title('Chief Executive Officer') == 'CEO'
        assert scraper._normalize_title('CEO') == 'CEO'
        assert scraper._normalize_title('President and CEO') == 'CEO'

    @patch('scrapers.form8k_scraper.get_config')
    @patch('scrapers.form8k_scraper.get_database')
    def test_normalize_cmo_titles(self, mock_get_db, mock_get_config):
        """Test CMO title normalization."""
        mock_config = Mock()
        mock_config.sec_user_agent = 'Test'
        mock_config.scraping = {}
        mock_config.sec_edgar = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = Form8KScraper(use_ai=False)

        assert scraper._normalize_title('Chief Medical Officer') == 'CMO'
        assert scraper._normalize_title('CMO') == 'CMO'


class TestDepartureReasonExtraction:
    """Tests for departure reason extraction."""

    def test_departure_patterns_defined(self):
        """Test departure reason patterns are defined."""
        expected_reasons = [
            'retirement', 'resignation', 'termination',
            'personal', 'mutual', 'health'
        ]

        for reason in expected_reasons:
            assert reason in Form8KScraper.DEPARTURE_PATTERNS

    @patch('scrapers.form8k_scraper.get_config')
    @patch('scrapers.form8k_scraper.get_database')
    def test_extract_retirement_reason(self, mock_get_db, mock_get_config):
        """Test extraction of retirement reason."""
        mock_config = Mock()
        mock_config.sec_user_agent = 'Test'
        mock_config.scraping = {}
        mock_config.sec_edgar = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = Form8KScraper(use_ai=False)

        text = "Dr. Smith announced his retirement after 15 years of service"
        reason = scraper._extract_reason(text.lower())
        assert reason == 'retirement'

    @patch('scrapers.form8k_scraper.get_config')
    @patch('scrapers.form8k_scraper.get_database')
    def test_extract_resignation_reason(self, mock_get_db, mock_get_config):
        """Test extraction of resignation reason."""
        mock_config = Mock()
        mock_config.sec_user_agent = 'Test'
        mock_config.scraping = {}
        mock_config.sec_edgar = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = Form8KScraper(use_ai=False)

        text = "The CEO resigned to pursue other opportunities"
        reason = scraper._extract_reason(text.lower())
        assert reason == 'resignation'


class TestVoluntaryDetermination:
    """Tests for voluntary/involuntary departure determination."""

    @patch('scrapers.form8k_scraper.get_config')
    @patch('scrapers.form8k_scraper.get_database')
    def test_retirement_is_voluntary(self, mock_get_db, mock_get_config):
        """Test that retirement is marked as voluntary."""
        mock_config = Mock()
        mock_config.sec_user_agent = 'Test'
        mock_config.scraping = {}
        mock_config.sec_edgar = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = Form8KScraper(use_ai=False)

        result = scraper._determine_if_voluntary("announced retirement", "Retirement")
        assert result is True

    @patch('scrapers.form8k_scraper.get_config')
    @patch('scrapers.form8k_scraper.get_database')
    def test_termination_is_involuntary(self, mock_get_db, mock_get_config):
        """Test that termination is marked as involuntary."""
        mock_config = Mock()
        mock_config.sec_user_agent = 'Test'
        mock_config.scraping = {}
        mock_config.sec_edgar = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = Form8KScraper(use_ai=False)

        result = scraper._determine_if_voluntary("terminated for cause", "Termination")
        assert result is False

    @patch('scrapers.form8k_scraper.get_config')
    @patch('scrapers.form8k_scraper.get_database')
    def test_pursue_other_is_voluntary(self, mock_get_db, mock_get_config):
        """Test that 'pursue other opportunities' is voluntary."""
        mock_config = Mock()
        mock_config.sec_user_agent = 'Test'
        mock_config.scraping = {}
        mock_config.sec_edgar = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = Form8KScraper(use_ai=False)

        result = scraper._determine_if_voluntary("leaving to pursue other opportunities", "Resignation")
        assert result is True


class TestSignalWeightCalculation:
    """Tests for signal weight calculation."""

    @patch('scrapers.form8k_scraper.get_config')
    @patch('scrapers.form8k_scraper.get_database')
    def test_ceo_departure_weight(self, mock_get_db, mock_get_config):
        """Test CEO departure signal weight calculation."""
        mock_config = Mock()
        mock_config.sec_user_agent = 'Test'
        mock_config.scraping = {}
        mock_config.sec_edgar = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = Form8KScraper(use_ai=False)

        weight, severity = scraper._calculate_signal_weight('CEO', 'Departure', True, "")

        assert weight == Form8KScraper.SIGNAL_WEIGHTS['CEO_DEPARTURE']
        assert severity >= 1 and severity <= 10

    @patch('scrapers.form8k_scraper.get_config')
    @patch('scrapers.form8k_scraper.get_database')
    def test_interim_appointment_adds_penalty(self, mock_get_db, mock_get_config):
        """Test that interim appointments add uncertainty penalty."""
        mock_config = Mock()
        mock_config.sec_user_agent = 'Test'
        mock_config.scraping = {}
        mock_config.sec_edgar = {}
        mock_config.anthropic_api_key = None
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = Mock()

        scraper = Form8KScraper(use_ai=False)

        weight_regular, _ = scraper._calculate_signal_weight('CEO', 'Appointment', None, "")
        weight_interim, _ = scraper._calculate_signal_weight('CEO', 'Appointment', None, "interim ceo appointed")

        # Interim should be lower (more negative adjustment)
        assert weight_interim < weight_regular


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
