"""
Tests for the validation module.
"""
import pytest
from datetime import date

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.validation import (
    TrialValidator,
    SignalValidator,
    CompanyValidator,
    PreprintValidator,
    PatentValidator,
    validate_trial,
    validate_signal,
    validate_company,
    validate_preprint,
    validate_patent,
    ValidationError,
    ValidationResult,
)


class TestTrialValidator:
    """Tests for TrialValidator."""

    def test_valid_trial(self, sample_trial_data):
        """Test validation of valid trial data."""
        result = validate_trial(sample_trial_data)
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.sanitized_data["trial_id"] == "NCT12345678"

    def test_invalid_trial_id_format(self):
        """Test validation fails for invalid NCT ID format."""
        data = {"trial_id": "INVALID123"}
        result = validate_trial(data)
        assert not result.is_valid
        assert any("Invalid NCT ID format" in str(e) for e in result.errors)

    def test_missing_trial_id(self):
        """Test validation fails for missing trial ID."""
        data = {"status": "RECRUITING"}
        result = validate_trial(data)
        assert not result.is_valid
        assert any("required" in str(e).lower() for e in result.errors)

    def test_trial_id_normalization(self):
        """Test that trial ID is normalized to uppercase."""
        data = {"trial_id": "nct12345678"}
        result = validate_trial(data)
        assert result.is_valid
        assert result.sanitized_data["trial_id"] == "NCT12345678"

    def test_unknown_status_warning(self):
        """Test warning for unknown status."""
        data = {"trial_id": "NCT12345678", "status": "UNKNOWN_STATUS"}
        result = validate_trial(data)
        assert result.is_valid  # Still valid, just warning
        assert any("Unknown status" in w for w in result.warnings)

    def test_negative_enrollment_error(self):
        """Test error for negative enrollment."""
        data = {"trial_id": "NCT12345678", "enrollment_target": -100}
        result = validate_trial(data)
        assert not result.is_valid
        assert any("negative" in str(e).lower() for e in result.errors)

    def test_date_parsing(self):
        """Test date parsing from string."""
        data = {
            "trial_id": "NCT12345678",
            "start_date": "2024-06-15",
            "expected_completion": date(2025, 12, 1),
        }
        result = validate_trial(data)
        assert result.is_valid
        assert result.sanitized_data["start_date"] == date(2024, 6, 15)
        assert result.sanitized_data["expected_completion"] == date(2025, 12, 1)


class TestSignalValidator:
    """Tests for SignalValidator."""

    def test_valid_signal(self, sample_signal_data):
        """Test validation of valid signal data."""
        result = validate_signal(sample_signal_data)
        assert result.is_valid
        assert result.sanitized_data["signal_type"] == "enrollment_increase"

    def test_missing_signal_type(self):
        """Test validation fails for missing signal type."""
        data = {"trial_id": "NCT12345678", "signal_weight": 2}
        result = validate_signal(data)
        assert not result.is_valid
        assert any("required" in str(e).lower() for e in result.errors)

    def test_invalid_signal_weight(self):
        """Test validation fails for out-of-range signal weight."""
        data = {
            "trial_id": "NCT12345678",
            "signal_type": "test_signal",
            "signal_weight": 10,  # Max is 5
        }
        result = validate_signal(data)
        assert not result.is_valid
        assert any("between -5 and 5" in str(e) for e in result.errors)

    def test_unknown_signal_type_warning(self):
        """Test warning for unknown signal type."""
        data = {
            "trial_id": "NCT12345678",
            "signal_type": "custom_signal_type",
            "signal_weight": 2,
        }
        result = validate_signal(data)
        assert result.is_valid  # Still valid
        assert any("Unknown signal type" in w for w in result.warnings)


class TestCompanyValidator:
    """Tests for CompanyValidator."""

    def test_valid_company(self, sample_company_data):
        """Test validation of valid company data."""
        result = validate_company(sample_company_data)
        assert result.is_valid
        assert result.sanitized_data["ticker"] == "MRNA"

    def test_invalid_ticker_format(self):
        """Test validation fails for invalid ticker."""
        data = {"ticker": "toolongticker", "company_name": "Test Co"}
        result = validate_company(data)
        assert not result.is_valid
        assert any("Invalid ticker format" in str(e) for e in result.errors)

    def test_ticker_normalization(self):
        """Test ticker is normalized to uppercase."""
        data = {"ticker": "mrna", "company_name": "Moderna Inc"}
        result = validate_company(data)
        assert result.is_valid
        assert result.sanitized_data["ticker"] == "MRNA"

    def test_missing_company_name(self):
        """Test validation fails for missing company name."""
        data = {"ticker": "MRNA"}
        result = validate_company(data)
        assert not result.is_valid

    def test_cik_zero_padding(self):
        """Test CIK is zero-padded."""
        data = {"ticker": "MRNA", "company_name": "Moderna", "cik": "1682852"}
        result = validate_company(data)
        assert result.is_valid
        assert result.sanitized_data["cik"] == "0001682852"


class TestPreprintValidator:
    """Tests for PreprintValidator."""

    def test_valid_pubmed_preprint(self, sample_preprint_data):
        """Test validation of valid PubMed data."""
        result = validate_preprint(sample_preprint_data)
        assert result.is_valid
        assert result.sanitized_data["source"] == "pubmed"

    def test_invalid_source(self):
        """Test validation fails for invalid source."""
        data = {"source": "unknown_source", "external_id": "12345"}
        result = validate_preprint(data)
        assert not result.is_valid
        assert any("Invalid source" in str(e) for e in result.errors)

    def test_missing_external_id(self):
        """Test validation fails for missing external ID."""
        data = {"source": "pubmed"}
        result = validate_preprint(data)
        assert not result.is_valid

    def test_medrxiv_source(self):
        """Test validation of medRxiv data."""
        data = {
            "source": "medrxiv",
            "external_id": "10.1101/2024.01.01.000001",
            "title": "Test Preprint",
        }
        result = validate_preprint(data)
        assert result.is_valid


class TestPatentValidator:
    """Tests for PatentValidator."""

    def test_valid_patent(self, sample_patent_data):
        """Test validation of valid patent data."""
        result = validate_patent(sample_patent_data)
        assert result.is_valid

    def test_missing_both_numbers(self):
        """Test validation fails when both patent and application numbers missing."""
        data = {"title": "Test Patent"}
        result = validate_patent(data)
        assert not result.is_valid
        assert any("required" in str(e).lower() for e in result.errors)

    def test_application_number_only(self):
        """Test validation passes with only application number."""
        data = {"application_number": "17/123,456", "title": "Test Patent"}
        result = validate_patent(data)
        assert result.is_valid

    def test_patent_number_cleanup(self):
        """Test patent number is cleaned up."""
        data = {
            "patent_number": "US 12,345,678",
            "title": "Test Patent",
        }
        result = validate_patent(data)
        assert result.is_valid
        assert result.sanitized_data["patent_number"] == "US12345678"


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_add_error_sets_invalid(self):
        """Test that adding an error sets is_valid to False."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        result.add_error("Test error", "test_field")
        assert not result.is_valid
        assert len(result.errors) == 1

    def test_add_warning_keeps_valid(self):
        """Test that adding a warning keeps is_valid True."""
        result = ValidationResult(is_valid=True)
        result.add_warning("Test warning")
        assert result.is_valid
        assert len(result.warnings) == 1
