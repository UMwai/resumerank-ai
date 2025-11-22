"""
Tests for the data extractors.
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch, MagicMock

from src.extractors.orange_book import OrangeBookExtractor
from src.extractors.uspto import USPTOExtractor, PatentExpirationCalculator
from src.extractors.fda_anda import ANDAExtractor, GenericCompetitionAnalyzer


class TestOrangeBookExtractor:
    """Tests for the OrangeBookExtractor class."""

    @pytest.fixture
    def extractor(self, tmp_path):
        return OrangeBookExtractor(
            cache_dir=str(tmp_path / "cache"),
            use_cache=False,
        )

    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor is not None
        assert not extractor.use_cache

    def test_cache_directory_creation(self, extractor):
        """Test that cache directory is created."""
        assert extractor.cache_dir.exists()

    @patch("requests.get")
    def test_download_failure_retry(self, mock_get, extractor):
        """Test retry logic on download failure."""
        mock_get.side_effect = Exception("Network error")

        with pytest.raises(Exception):
            extractor._download_orange_book()

    def test_top_pharma_companies_list(self, extractor):
        """Test that top pharma companies are defined."""
        assert len(extractor.TOP_PHARMA_COMPANIES) > 0
        assert "ABBVIE" in extractor.TOP_PHARMA_COMPANIES


class TestUSPTOExtractor:
    """Tests for the USPTOExtractor class."""

    @pytest.fixture
    def extractor(self):
        return USPTOExtractor()

    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor is not None
        assert extractor.api_key is None

    def test_rate_limit_setting(self, extractor):
        """Test rate limit is configured."""
        assert extractor.RATE_LIMIT == 45

    def test_patent_term_constant(self, extractor):
        """Test patent term constant."""
        assert extractor.MODERN_PATENT_TERM_YEARS == 20


class TestPatentExpirationCalculator:
    """Tests for the PatentExpirationCalculator class."""

    def test_basic_expiration_from_filing(self):
        """Test basic 20-year expiration calculation."""
        calculator = PatentExpirationCalculator()
        result = calculator.calculate_expiration(
            filing_date=date(2010, 6, 15)
        )

        assert result["base_expiration_date"] == date(2030, 6, 15)
        assert result["final_expiration_date"] == date(2030, 6, 15)

    def test_expiration_with_pta(self):
        """Test expiration with Patent Term Adjustment."""
        calculator = PatentExpirationCalculator()
        result = calculator.calculate_expiration(
            filing_date=date(2010, 6, 15),
            pta_days=180,  # 6 months
        )

        # Should be 180 days after base
        assert result["pta_adjusted_date"] > result["base_expiration_date"]

    def test_expiration_with_pte(self):
        """Test expiration with Patent Term Extension."""
        calculator = PatentExpirationCalculator()
        result = calculator.calculate_expiration(
            filing_date=date(2010, 6, 15),
            pte_days=365,  # 1 year
        )

        assert result["pte_adjusted_date"] > result["base_expiration_date"]

    def test_expiration_with_pediatric(self):
        """Test expiration with pediatric exclusivity."""
        calculator = PatentExpirationCalculator()
        result = calculator.calculate_expiration(
            filing_date=date(2010, 6, 15),
            pediatric_extension_months=6,
        )

        # Should be 6 months after base
        expected = date(2030, 12, 15)
        assert result["pediatric_exclusivity_date"] == expected

    def test_expiration_orange_book_override(self):
        """Test that Orange Book expiration takes precedence."""
        calculator = PatentExpirationCalculator()
        ob_date = date(2025, 12, 31)

        result = calculator.calculate_expiration(
            filing_date=date(2010, 6, 15),
            orange_book_expiration=ob_date,
        )

        assert result["base_expiration_date"] == ob_date

    def test_days_until_expiration_future(self):
        """Test days calculation for future date."""
        future_date = date.today().replace(year=date.today().year + 1)
        days = PatentExpirationCalculator.days_until_expiration(future_date)
        assert days > 0

    def test_days_until_expiration_past(self):
        """Test days calculation for past date."""
        past_date = date(2020, 1, 1)
        days = PatentExpirationCalculator.days_until_expiration(past_date)
        assert days < 0

    def test_is_expired_true(self):
        """Test is_expired for past date."""
        past_date = date(2020, 1, 1)
        assert PatentExpirationCalculator.is_expired(past_date) is True

    def test_is_expired_false(self):
        """Test is_expired for future date."""
        future_date = date.today().replace(year=date.today().year + 5)
        assert PatentExpirationCalculator.is_expired(future_date) is False


class TestANDAExtractor:
    """Tests for the ANDAExtractor class."""

    @pytest.fixture
    def extractor(self, tmp_path):
        return ANDAExtractor(cache_dir=str(tmp_path / "anda_cache"))

    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor is not None
        assert extractor.cache_dir.exists()

    def test_generic_companies_mapping(self, extractor):
        """Test generic companies ticker mapping."""
        assert extractor.get_generic_company_ticker("TEVA PHARMA") == "TEVA"
        assert extractor.get_generic_company_ticker("MYLAN LABS") == "VTRS"
        assert extractor.get_generic_company_ticker("Unknown Corp") is None


class TestGenericCompetitionAnalyzer:
    """Tests for the GenericCompetitionAnalyzer class."""

    def test_erosion_one_generic(self):
        """Test erosion with single generic competitor."""
        result = GenericCompetitionAnalyzer.estimate_revenue_erosion(
            num_generic_competitors=1,
            annual_revenue=1_000_000_000,
        )

        assert result["erosion_rate"] == 0.50
        assert result["revenue_loss"] == 500_000_000

    def test_erosion_multiple_generics(self):
        """Test erosion with multiple generic competitors."""
        result = GenericCompetitionAnalyzer.estimate_revenue_erosion(
            num_generic_competitors=5,
            annual_revenue=1_000_000_000,
        )

        assert result["erosion_rate"] == 0.80  # 4+ generics rate
        assert result["revenue_loss"] == 800_000_000

    def test_classify_blockbuster(self):
        """Test blockbuster classification."""
        tier = GenericCompetitionAnalyzer.classify_opportunity(1_500_000_000)
        assert tier == "BLOCKBUSTER"

    def test_classify_high_value(self):
        """Test high value classification."""
        tier = GenericCompetitionAnalyzer.classify_opportunity(750_000_000)
        assert tier == "HIGH_VALUE"

    def test_classify_medium_value(self):
        """Test medium value classification."""
        tier = GenericCompetitionAnalyzer.classify_opportunity(200_000_000)
        assert tier == "MEDIUM_VALUE"

    def test_classify_small(self):
        """Test small classification."""
        tier = GenericCompetitionAnalyzer.classify_opportunity(50_000_000)
        assert tier == "SMALL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
