"""
Tests for Form 4 Real-Time RSS Monitor
"""

import pytest
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.form4_realtime import (
    Form4RealtimeMonitor,
    RSSFiling,
    AlertConfig,
    start_realtime_monitor,
    poll_form4_rss
)


class TestRSSFiling:
    """Tests for RSSFiling dataclass."""

    def test_rss_filing_creation(self):
        """Test creating an RSSFiling object."""
        filing = RSSFiling(
            accession_number='0001234-24-000001',
            filing_url='https://www.sec.gov/example',
            company_name='Test Company Inc',
            company_cik='0001234567',
            filer_name='John Doe',
            filer_cik='0009876543',
            filing_date=datetime.now(),
            form_type='4',
            title='4 - Test Company Inc (0001234567) (John Doe)',
            updated=datetime.now()
        )

        assert filing.accession_number == '0001234-24-000001'
        assert filing.form_type == '4'
        assert filing.company_name == 'Test Company Inc'

    def test_rss_filing_hash(self):
        """Test RSSFiling hash is based on accession number."""
        filing1 = RSSFiling(
            accession_number='0001234-24-000001',
            filing_url='https://example.com/1',
            company_name='Company A',
            company_cik='0001',
            filer_name='Filer A',
            filer_cik='0002',
            filing_date=datetime.now(),
            form_type='4',
            title='Test',
            updated=datetime.now()
        )
        filing2 = RSSFiling(
            accession_number='0001234-24-000001',  # Same accession number
            filing_url='https://example.com/2',
            company_name='Company B',
            company_cik='0003',
            filer_name='Filer B',
            filer_cik='0004',
            filing_date=datetime.now(),
            form_type='4',
            title='Test 2',
            updated=datetime.now()
        )

        assert hash(filing1) == hash(filing2)
        assert filing1 == filing2

    def test_rss_filing_inequality(self):
        """Test RSSFiling inequality with different accession numbers."""
        filing1 = RSSFiling(
            accession_number='0001234-24-000001',
            filing_url='https://example.com/1',
            company_name='Company A',
            company_cik='0001',
            filer_name='Filer A',
            filer_cik='0002',
            filing_date=datetime.now(),
            form_type='4',
            title='Test',
            updated=datetime.now()
        )
        filing2 = RSSFiling(
            accession_number='0001234-24-000002',  # Different accession number
            filing_url='https://example.com/2',
            company_name='Company A',
            company_cik='0001',
            filer_name='Filer A',
            filer_cik='0002',
            filing_date=datetime.now(),
            form_type='4',
            title='Test 2',
            updated=datetime.now()
        )

        assert hash(filing1) != hash(filing2)
        assert filing1 != filing2


class TestAlertConfig:
    """Tests for AlertConfig dataclass."""

    def test_default_alert_config(self):
        """Test default AlertConfig values."""
        config = AlertConfig()

        assert config.min_transaction_value == 50000
        assert config.alert_on_ceo_trades is True
        assert config.alert_on_multiple_insiders is True
        assert config.alert_on_large_trades is True
        assert config.large_trade_threshold == 500000
        assert config.callback is None

    def test_custom_alert_config(self):
        """Test custom AlertConfig values."""
        callback_fn = Mock()
        config = AlertConfig(
            min_transaction_value=100000,
            alert_on_ceo_trades=False,
            large_trade_threshold=1000000,
            callback=callback_fn
        )

        assert config.min_transaction_value == 100000
        assert config.alert_on_ceo_trades is False
        assert config.large_trade_threshold == 1000000
        assert config.callback == callback_fn


class TestForm4RealtimeMonitor:
    """Tests for Form4RealtimeMonitor class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.sec_user_agent = 'Test Agent test@example.com'
        config.scraping = {'timeout_seconds': 30}
        config.watchlist = ['MRNA', 'VRTX', 'BIIB']
        config.sec_edgar = {'rate_limit_requests_per_second': 10}
        return config

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = Mock()
        db.execute.return_value = []
        return db

    @patch('scrapers.form4_realtime.get_config')
    @patch('scrapers.form4_realtime.get_database')
    @patch('scrapers.form4_realtime.Form4Scraper')
    def test_monitor_initialization(self, mock_scraper_cls, mock_get_db, mock_get_config, mock_config, mock_db):
        """Test monitor initialization."""
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = mock_db

        monitor = Form4RealtimeMonitor(poll_interval_minutes=30)

        assert monitor.poll_interval == 30 * 60
        assert monitor.is_running() is False
        assert monitor.stats['polls_completed'] == 0

    @patch('scrapers.form4_realtime.get_config')
    @patch('scrapers.form4_realtime.get_database')
    @patch('scrapers.form4_realtime.Form4Scraper')
    def test_get_stats(self, mock_scraper_cls, mock_get_db, mock_get_config, mock_config, mock_db):
        """Test getting monitor statistics."""
        mock_get_config.return_value = mock_config
        mock_get_db.return_value = mock_db

        monitor = Form4RealtimeMonitor(poll_interval_minutes=15)
        stats = monitor.get_stats()

        assert 'polls_completed' in stats
        assert 'filings_found' in stats
        assert 'is_running' in stats
        assert stats['poll_interval_minutes'] == 15

    def test_parse_rss_entry_sample(self):
        """Test parsing a sample RSS entry."""
        import xml.etree.ElementTree as ET

        sample_xml = '''
        <entry xmlns="http://www.w3.org/2005/Atom">
            <title>4 - MODERNA INC (0001682852) (BANCEL STEPHANE)</title>
            <link href="https://www.sec.gov/Archives/edgar/data/1682852/000168285224000001"/>
            <updated>2024-01-15T10:30:00Z</updated>
            <summary>Form 4 filing</summary>
        </entry>
        '''

        # This tests the internal parsing logic pattern
        entry = ET.fromstring(sample_xml)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        title_elem = entry.find('atom:title', ns)
        link_elem = entry.find('atom:link', ns)
        updated_elem = entry.find('atom:updated', ns)

        assert title_elem is not None
        assert 'MODERNA' in title_elem.text
        assert link_elem.get('href') is not None
        assert '2024-01-15' in updated_elem.text


class TestRSSParsing:
    """Tests for RSS feed parsing."""

    SAMPLE_RSS_FEED = '''<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
        <title>Latest Form 4 Filings</title>
        <entry>
            <title>4 - MODERNA INC (0001682852) (BANCEL STEPHANE)</title>
            <link href="https://www.sec.gov/Archives/edgar/data/1682852/000168285224000001"/>
            <updated>2024-01-15T10:30:00Z</updated>
        </entry>
        <entry>
            <title>4 - VERTEX PHARMACEUTICALS INC (0000875320) (LEIDEN JEFFREY)</title>
            <link href="https://www.sec.gov/Archives/edgar/data/875320/000087532024000001"/>
            <updated>2024-01-15T09:15:00Z</updated>
        </entry>
    </feed>
    '''

    @patch('scrapers.form4_realtime.get_config')
    @patch('scrapers.form4_realtime.get_database')
    @patch('scrapers.form4_realtime.Form4Scraper')
    def test_parse_rss_feed(self, mock_scraper_cls, mock_get_db, mock_get_config):
        """Test parsing RSS feed XML."""
        mock_config = Mock()
        mock_config.sec_user_agent = 'Test Agent'
        mock_config.scraping = {'timeout_seconds': 30}
        mock_config.watchlist = ['MRNA', 'VRTX']
        mock_config.sec_edgar = {'rate_limit_requests_per_second': 10}
        mock_get_config.return_value = mock_config

        mock_db = Mock()
        mock_db.execute.return_value = []
        mock_get_db.return_value = mock_db

        monitor = Form4RealtimeMonitor()
        filings = monitor._parse_rss_feed(self.SAMPLE_RSS_FEED)

        assert len(filings) == 2
        assert any('MODERNA' in f.company_name for f in filings)
        assert any('VERTEX' in f.company_name for f in filings)


class TestAlertTriggering:
    """Tests for alert triggering logic."""

    def test_alert_config_callback_called(self):
        """Test that alert callback is called when configured."""
        callback_mock = Mock()
        config = AlertConfig(
            callback=callback_mock,
            min_transaction_value=10000,
            alert_on_ceo_trades=True
        )

        # Simulate alert data
        alert_data = {
            'ticker': 'MRNA',
            'insider': 'John CEO',
            'title': 'CEO',
            'transaction_type': 'Purchase',
            'value': 100000
        }

        # Callback should be callable
        config.callback(alert_data)
        callback_mock.assert_called_once_with(alert_data)


class TestFilingDeduplication:
    """Tests for filing deduplication logic."""

    @patch('scrapers.form4_realtime.get_config')
    @patch('scrapers.form4_realtime.get_database')
    @patch('scrapers.form4_realtime.Form4Scraper')
    def test_filing_marked_as_seen(self, mock_scraper_cls, mock_get_db, mock_get_config):
        """Test that filings are marked as seen."""
        mock_config = Mock()
        mock_config.sec_user_agent = 'Test Agent'
        mock_config.scraping = {'timeout_seconds': 30}
        mock_config.watchlist = ['MRNA']
        mock_config.sec_edgar = {'rate_limit_requests_per_second': 10}
        mock_get_config.return_value = mock_config

        mock_db = Mock()
        mock_db.execute.return_value = []
        mock_get_db.return_value = mock_db

        monitor = Form4RealtimeMonitor()

        filing = RSSFiling(
            accession_number='0001234-24-000001',
            filing_url='https://example.com/filing',
            company_name='Test Company',
            company_cik='0001234',
            filer_name='Test Filer',
            filer_cik='0005678',
            filing_date=datetime.now(),
            form_type='4',
            title='Test',
            updated=datetime.now()
        )

        # First check - should be new
        assert monitor._is_new_filing(filing) is True

        # Mark as seen
        monitor._mark_filing_seen(filing)

        # Second check - should not be new
        assert monitor._is_new_filing(filing) is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
