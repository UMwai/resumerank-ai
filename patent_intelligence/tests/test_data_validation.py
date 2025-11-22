"""
Tests for the data validation module.
"""

import pytest
from datetime import date

from src.utils.data_validation import (
    DataValidator,
    ValidationResult,
    FieldRule,
    DuplicateDetector,
    DataQualityReport,
    validate_required_fields,
    validate_date_range,
    calculate_field_completeness,
)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_create_valid_result(self):
        """Test creating a valid result."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_add_error(self):
        """Test adding an error."""
        result = ValidationResult(is_valid=True)
        result.add_error("Test error", "test_field")

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "test_field" in result.field_errors

    def test_add_warning(self):
        """Test adding a warning."""
        result = ValidationResult(is_valid=True)
        result.add_warning("Test warning")

        assert result.is_valid is True  # Warnings don't affect validity
        assert len(result.warnings) == 1

    def test_merge_results(self):
        """Test merging validation results."""
        result1 = ValidationResult(is_valid=True)
        result1.add_warning("Warning 1")

        result2 = ValidationResult(is_valid=True)
        result2.add_error("Error 1", "field1")

        result1.merge(result2)

        assert result1.is_valid is False
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 1


class TestDataValidator:
    """Tests for DataValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return DataValidator()

    def test_validate_drug_valid(self, validator):
        """Test validating a valid drug record."""
        drug = {
            "brand_name": "Humira",
            "generic_name": "adalimumab",
            "nda_number": "125057",
            "branded_company": "AbbVie Inc.",
            "market_status": "ACTIVE",
        }
        result = validator.validate_drug(drug)
        assert result.is_valid is True

    def test_validate_drug_missing_brand_name(self, validator):
        """Test validating drug with missing brand name."""
        drug = {
            "generic_name": "adalimumab",
            "nda_number": "125057",
        }
        result = validator.validate_drug(drug)
        assert result.is_valid is False
        assert "brand_name" in result.field_errors

    def test_validate_drug_invalid_market_status(self, validator):
        """Test validating drug with invalid market status."""
        drug = {
            "brand_name": "Humira",
            "market_status": "INVALID_STATUS",
        }
        result = validator.validate_drug(drug)
        assert result.is_valid is False

    def test_validate_patent_valid(self, validator):
        """Test validating a valid patent record."""
        patent = {
            "patent_number": "6090382",
            "patent_type": "COMPOSITION",
            "base_expiration_date": date(2024, 12, 31),
            "patent_status": "ACTIVE",
        }
        result = validator.validate_patent(patent)
        assert result.is_valid is True

    def test_validate_patent_missing_number(self, validator):
        """Test validating patent with missing number."""
        patent = {
            "patent_type": "COMPOSITION",
            "base_expiration_date": date(2024, 12, 31),
        }
        result = validator.validate_patent(patent)
        assert result.is_valid is False

    def test_validate_patent_invalid_type(self, validator):
        """Test validating patent with invalid type."""
        patent = {
            "patent_number": "6090382",
            "patent_type": "INVALID_TYPE",
            "base_expiration_date": date(2024, 12, 31),
        }
        result = validator.validate_patent(patent)
        assert result.is_valid is False

    def test_validate_litigation_valid(self, validator):
        """Test validating a valid litigation record."""
        litigation = {
            "case_id": "case123",
            "case_name": "Test Corp v. Generic Co",
            "case_type": "PATENT",
            "outcome": "ONGOING",
        }
        result = validator.validate_litigation(litigation)
        assert result.is_valid is True

    def test_validate_litigation_missing_case_id(self, validator):
        """Test validating litigation with missing case ID."""
        litigation = {
            "case_name": "Test Corp v. Generic Co",
        }
        result = validator.validate_litigation(litigation)
        assert result.is_valid is False

    def test_validate_anda_valid(self, validator):
        """Test validating a valid ANDA record."""
        anda = {
            "anda_number": "123456",
            "generic_company": "Teva Pharmaceutical",
            "status": "APPROVED",
        }
        result = validator.validate_anda(anda)
        assert result.is_valid is True

    def test_validate_anda_missing_company(self, validator):
        """Test validating ANDA with missing company."""
        anda = {
            "anda_number": "123456",
        }
        result = validator.validate_anda(anda)
        assert result.is_valid is False

    def test_validate_batch(self, validator):
        """Test batch validation."""
        records = [
            {"brand_name": "Drug A", "market_status": "ACTIVE"},
            {"brand_name": "", "market_status": "ACTIVE"},  # Invalid
            {"brand_name": "Drug C", "market_status": "ACTIVE"},
        ]
        valid, invalid = validator.validate_batch(records, "drug")

        assert len(valid) == 2
        assert len(invalid) == 1


class TestDuplicateDetector:
    """Tests for DuplicateDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return DuplicateDetector()

    def test_detect_drug_duplicates(self, detector):
        """Test detecting duplicate drugs."""
        records = [
            {"brand_name": "Humira", "nda_number": "125057"},
            {"brand_name": "Humira", "nda_number": "125057"},  # Duplicate
            {"brand_name": "Keytruda", "nda_number": "125514"},
        ]
        unique, duplicates = detector.detect_duplicates(records, "drug")

        assert len(unique) == 2
        assert len(duplicates) == 1

    def test_detect_patent_duplicates(self, detector):
        """Test detecting duplicate patents."""
        records = [
            {"patent_number": "6090382", "patent_type": "COMPOSITION"},
            {"patent_number": "6090382", "patent_type": "METHOD_OF_USE"},  # Duplicate number
            {"patent_number": "7654321", "patent_type": "FORMULATION"},
        ]
        unique, duplicates = detector.detect_duplicates(records, "patent")

        assert len(unique) == 2
        assert len(duplicates) == 1

    def test_detect_litigation_duplicates(self, detector):
        """Test detecting duplicate litigation records."""
        records = [
            {"case_id": "case001", "court": "Delaware"},
            {"case_id": "case001", "court": "Delaware"},  # Duplicate
            {"case_id": "case002", "court": "New Jersey"},
        ]
        unique, duplicates = detector.detect_duplicates(records, "litigation")

        assert len(unique) == 2
        assert len(duplicates) == 1

    def test_clear_seen_keys(self, detector):
        """Test clearing seen keys."""
        records = [{"patent_number": "6090382"}]
        detector.detect_duplicates(records, "patent")

        detector.clear("patent")

        # Should now see the same record as unique
        unique, duplicates = detector.detect_duplicates(records, "patent")
        assert len(unique) == 1
        assert len(duplicates) == 0


class TestDataQualityReport:
    """Tests for DataQualityReport class."""

    @pytest.fixture
    def report(self):
        """Create a report instance."""
        return DataQualityReport()

    def test_add_validation_results(self, report):
        """Test adding validation results."""
        report.add_validation_results(
            record_type="drug",
            total=100,
            valid=95,
            invalid=5,
            errors=["Missing brand name", "Invalid status"],
        )

        assert "drug" in report.metrics
        assert report.metrics["drug"]["validation"]["total"] == 100
        assert report.metrics["drug"]["validation"]["validity_rate"] == 0.95

    def test_add_duplicate_results(self, report):
        """Test adding duplicate detection results."""
        report.add_duplicate_results(
            record_type="patent",
            total=100,
            unique=90,
            duplicates=10,
        )

        assert "patent" in report.metrics
        assert report.metrics["patent"]["duplicates"]["duplicate_rate"] == 0.10

    def test_generate_report(self, report):
        """Test generating text report."""
        report.add_validation_results(
            record_type="drug",
            total=100,
            valid=95,
            invalid=5,
            errors=["Test error"],
        )

        text = report.generate_report()

        assert "Data Quality Report" in text
        assert "drug" in text.lower()

    def test_to_dict(self, report):
        """Test converting to dictionary."""
        report.add_validation_results(
            record_type="drug",
            total=100,
            valid=95,
            invalid=5,
            errors=[],
        )

        result = report.to_dict()
        assert isinstance(result, dict)
        assert "drug" in result


class TestValidationHelperFunctions:
    """Tests for validation helper functions."""

    def test_validate_required_fields_all_present(self):
        """Test validating when all required fields are present."""
        record = {
            "field1": "value1",
            "field2": "value2",
        }
        result = validate_required_fields(record, ["field1", "field2"])
        assert result.is_valid is True

    def test_validate_required_fields_missing(self):
        """Test validating when required fields are missing."""
        record = {
            "field1": "value1",
        }
        result = validate_required_fields(record, ["field1", "field2"])
        assert result.is_valid is False

    def test_validate_required_fields_empty_value(self):
        """Test validating when required field is empty."""
        record = {
            "field1": "",
            "field2": "value2",
        }
        result = validate_required_fields(record, ["field1", "field2"])
        assert result.is_valid is False

    def test_validate_date_range_valid(self):
        """Test validating date within range."""
        result = validate_date_range(
            date_value=date(2023, 6, 15),
            min_date=date(2023, 1, 1),
            max_date=date(2023, 12, 31),
        )
        assert result.is_valid is True

    def test_validate_date_range_before_min(self):
        """Test validating date before minimum."""
        result = validate_date_range(
            date_value=date(2022, 6, 15),
            min_date=date(2023, 1, 1),
            max_date=date(2023, 12, 31),
        )
        assert result.is_valid is False

    def test_validate_date_range_after_max(self):
        """Test validating date after maximum."""
        result = validate_date_range(
            date_value=date(2024, 6, 15),
            min_date=date(2023, 1, 1),
            max_date=date(2023, 12, 31),
        )
        assert result.is_valid is False

    def test_calculate_field_completeness(self):
        """Test calculating field completeness."""
        records = [
            {"field1": "a", "field2": "b", "field3": None},
            {"field1": "c", "field2": None, "field3": "d"},
            {"field1": "e", "field2": "f", "field3": "g"},
        ]
        completeness = calculate_field_completeness(
            records,
            ["field1", "field2", "field3"],
        )

        assert completeness["field1"] == 1.0  # 3/3
        assert completeness["field2"] == 2/3
        assert completeness["field3"] == 2/3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
