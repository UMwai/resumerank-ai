"""
Tests for the Scrapers
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestForm4Scraper:
    """Tests for Form 4 scraper."""

    def test_transaction_codes_defined(self):
        """Ensure all transaction codes are defined."""
        from scrapers.form4_scraper import Form4Scraper

        expected_codes = ['P', 'S', 'A', 'D', 'F', 'M', 'G', 'W']

        for code in expected_codes:
            assert code in Form4Scraper.TRANSACTION_CODES

    def test_purchase_code(self):
        """Test purchase transaction code."""
        from scrapers.form4_scraper import Form4Scraper

        assert Form4Scraper.TRANSACTION_CODES['P'] == 'Purchase'

    def test_sale_code(self):
        """Test sale transaction code."""
        from scrapers.form4_scraper import Form4Scraper

        assert Form4Scraper.TRANSACTION_CODES['S'] == 'Sale'


class TestJobScraper:
    """Tests for Job scraper."""

    def test_department_classification_commercial(self):
        """Test commercial job classification."""
        from scrapers.job_scraper import JobScraper

        scraper = JobScraper.__new__(JobScraper)
        scraper.config = Mock()
        scraper.config.scraping = {}

        result = scraper.classify_job("Sales Representative")
        assert result['department'] == 'Commercial'
        assert result['is_commercial_role'] is True

    def test_department_classification_rd(self):
        """Test R&D job classification."""
        from scrapers.job_scraper import JobScraper

        scraper = JobScraper.__new__(JobScraper)
        scraper.config = Mock()
        scraper.config.scraping = {}

        result = scraper.classify_job("Research Scientist")
        assert result['department'] == 'R&D'
        assert result['is_rd_role'] is True

    def test_department_classification_clinical(self):
        """Test clinical job classification."""
        from scrapers.job_scraper import JobScraper

        scraper = JobScraper.__new__(JobScraper)
        scraper.config = Mock()
        scraper.config.scraping = {}

        result = scraper.classify_job("Clinical Research Associate")
        assert result['department'] == 'Clinical'
        assert result['is_clinical_role'] is True

    def test_department_classification_manufacturing(self):
        """Test manufacturing job classification."""
        from scrapers.job_scraper import JobScraper

        scraper = JobScraper.__new__(JobScraper)
        scraper.config = Mock()
        scraper.config.scraping = {}

        result = scraper.classify_job("Manufacturing Engineer")
        assert result['department'] == 'Manufacturing'
        assert result['is_manufacturing_role'] is True

    def test_seniority_vp(self):
        """Test VP seniority detection."""
        from scrapers.job_scraper import JobScraper

        scraper = JobScraper.__new__(JobScraper)
        scraper.config = Mock()
        scraper.config.scraping = {}

        result = scraper.classify_job("Vice President of Sales")
        assert result['seniority_level'] == 'VP'
        assert result['is_senior_role'] is True

    def test_seniority_director(self):
        """Test Director seniority detection."""
        from scrapers.job_scraper import JobScraper

        scraper = JobScraper.__new__(JobScraper)
        scraper.config = Mock()
        scraper.config.scraping = {}

        result = scraper.classify_job("Director of Clinical Operations")
        assert result['seniority_level'] == 'Director'
        assert result['is_senior_role'] is True

    def test_seniority_entry(self):
        """Test entry level detection."""
        from scrapers.job_scraper import JobScraper

        scraper = JobScraper.__new__(JobScraper)
        scraper.config = Mock()
        scraper.config.scraping = {}

        result = scraper.classify_job("Junior Research Associate")
        assert result['seniority_level'] == 'Entry'
        assert result['is_senior_role'] is False

    def test_msl_role_high_weight(self):
        """Test MSL roles get high signal weight."""
        from scrapers.job_scraper import JobScraper

        scraper = JobScraper.__new__(JobScraper)
        scraper.config = Mock()
        scraper.config.scraping = {}

        result = scraper.classify_job("Medical Science Liaison")
        assert result['signal_weight'] > 0

    def test_commercial_buildup_weight(self):
        """Test commercial roles get positive weight."""
        from scrapers.job_scraper import JobScraper

        scraper = JobScraper.__new__(JobScraper)
        scraper.config = Mock()
        scraper.config.scraping = {}

        result = scraper.classify_job("Territory Sales Manager")
        assert result['signal_weight'] > 0


class TestForm13FScraper:
    """Tests for 13F scraper."""

    def test_top_funds_defined(self):
        """Ensure top biotech funds are defined."""
        from scrapers.form13f_scraper import Form13FScraper

        assert len(Form13FScraper.TOP_BIOTECH_FUNDS) == 10

        # Check for key funds
        fund_names = [f['name'] for f in Form13FScraper.TOP_BIOTECH_FUNDS]
        assert 'Baker Bros Advisors LP' in fund_names
        assert 'RA Capital Management LP' in fund_names
        assert 'Perceptive Advisors LLC' in fund_names

    def test_signal_weights_defined(self):
        """Ensure signal weights are defined."""
        from scrapers.form13f_scraper import Form13FScraper

        expected_signals = [
            'NEW_POSITION', 'INCREASE_GT_50', 'DECREASE_GT_50', 'EXIT_POSITION'
        ]

        for signal in expected_signals:
            assert signal in Form13FScraper.SIGNAL_WEIGHTS

    def test_new_position_positive(self):
        """Test new position signal is positive."""
        from scrapers.form13f_scraper import Form13FScraper

        assert Form13FScraper.SIGNAL_WEIGHTS['NEW_POSITION'] > 0

    def test_exit_position_negative(self):
        """Test exit position signal is negative."""
        from scrapers.form13f_scraper import Form13FScraper

        assert Form13FScraper.SIGNAL_WEIGHTS['EXIT_POSITION'] < 0


class TestCompanyCareerUrls:
    """Test company career URL configurations."""

    def test_career_urls_defined(self):
        """Ensure career URLs are defined for key companies."""
        from scrapers.job_scraper import JobScraper

        expected_companies = ['MRNA', 'VRTX', 'CRSP', 'BEAM']

        for ticker in expected_companies:
            assert ticker in JobScraper.COMPANY_CAREER_URLS

    def test_greenhouse_url_format(self):
        """Test Greenhouse URLs are properly formatted."""
        from scrapers.job_scraper import JobScraper

        mrna_url = JobScraper.COMPANY_CAREER_URLS.get('MRNA', '')
        assert 'greenhouse.io' in mrna_url


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
