"""
Data Validation Module

Provides comprehensive data validation for the Patent Intelligence system.
Implements validation rules, anomaly detection, and data quality checks.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    field_errors: Dict[str, List[str]] = field(default_factory=dict)

    def add_error(self, message: str, field: Optional[str] = None) -> None:
        """Add an error message."""
        self.errors.append(message)
        if field:
            if field not in self.field_errors:
                self.field_errors[field] = []
            self.field_errors[field].append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        for field, errors in other.field_errors.items():
            if field not in self.field_errors:
                self.field_errors[field] = []
            self.field_errors[field].extend(errors)


@dataclass
class FieldRule:
    """Validation rule for a field."""
    field_name: str
    required: bool = False
    field_type: Optional[type] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[Set[str]] = None
    min_value: Optional[Union[int, float, date]] = None
    max_value: Optional[Union[int, float, date]] = None
    custom_validator: Optional[Callable[[Any], Tuple[bool, str]]] = None


class DataValidator:
    """
    Validates data records for the Patent Intelligence system.

    Provides validation for:
    - Drug records
    - Patent records
    - Litigation records
    - ANDA records
    - Calendar events
    """

    # Valid patent types
    VALID_PATENT_TYPES = {"COMPOSITION", "METHOD_OF_USE", "FORMULATION", "PROCESS", "OTHER"}

    # Valid patent statuses
    VALID_PATENT_STATUSES = {"ACTIVE", "EXPIRED", "INVALIDATED", "DELISTED"}

    # Valid case types
    VALID_CASE_TYPES = {"PATENT", "ANDA", "IPR", "PGR"}

    # Valid litigation outcomes
    VALID_OUTCOMES = {
        "PATENT_UPHELD", "PATENT_INVALIDATED", "SETTLED",
        "ONGOING", "DISMISSED", "PENDING"
    }

    # Valid ANDA statuses
    VALID_ANDA_STATUSES = {"PENDING", "TENTATIVE", "APPROVED", "WITHDRAWN"}

    # Valid market statuses
    VALID_MARKET_STATUSES = {"ACTIVE", "DISCONTINUED", "WITHDRAWN"}

    # Patent number pattern
    PATENT_NUMBER_PATTERN = r"^(RE)?\d{5,8}$"

    # NDA number pattern
    NDA_NUMBER_PATTERN = r"^\d{5,6}$"

    def __init__(self):
        """Initialize the validator."""
        self._drug_rules = self._build_drug_rules()
        self._patent_rules = self._build_patent_rules()
        self._litigation_rules = self._build_litigation_rules()
        self._anda_rules = self._build_anda_rules()

    def _build_drug_rules(self) -> List[FieldRule]:
        """Build validation rules for drug records."""
        return [
            FieldRule(
                field_name="brand_name",
                required=True,
                field_type=str,
                min_length=1,
                max_length=200,
            ),
            FieldRule(
                field_name="generic_name",
                required=False,
                field_type=str,
                max_length=200,
            ),
            FieldRule(
                field_name="nda_number",
                required=False,
                field_type=str,
                pattern=self.NDA_NUMBER_PATTERN,
            ),
            FieldRule(
                field_name="branded_company",
                required=False,
                field_type=str,
                max_length=200,
            ),
            FieldRule(
                field_name="annual_revenue",
                required=False,
                field_type=int,
                min_value=0,
                max_value=100_000_000_000,  # $100B max
            ),
            FieldRule(
                field_name="fda_approval_date",
                required=False,
                field_type=date,
                min_value=date(1900, 1, 1),
                max_value=date.today(),
            ),
            FieldRule(
                field_name="market_status",
                required=False,
                field_type=str,
                allowed_values=self.VALID_MARKET_STATUSES,
            ),
        ]

    def _build_patent_rules(self) -> List[FieldRule]:
        """Build validation rules for patent records."""
        return [
            FieldRule(
                field_name="patent_number",
                required=True,
                field_type=str,
                pattern=self.PATENT_NUMBER_PATTERN,
            ),
            FieldRule(
                field_name="patent_type",
                required=False,
                field_type=str,
                allowed_values=self.VALID_PATENT_TYPES,
            ),
            FieldRule(
                field_name="patent_status",
                required=False,
                field_type=str,
                allowed_values=self.VALID_PATENT_STATUSES,
            ),
            FieldRule(
                field_name="base_expiration_date",
                required=True,
                field_type=date,
                min_value=date(1990, 1, 1),
                max_value=date(2100, 1, 1),
            ),
            FieldRule(
                field_name="pta_days",
                required=False,
                field_type=int,
                min_value=0,
                max_value=3650,  # Max ~10 years
            ),
            FieldRule(
                field_name="pte_days",
                required=False,
                field_type=int,
                min_value=0,
                max_value=1825,  # Max 5 years
            ),
        ]

    def _build_litigation_rules(self) -> List[FieldRule]:
        """Build validation rules for litigation records."""
        return [
            FieldRule(
                field_name="case_id",
                required=True,
                field_type=str,
                min_length=1,
                max_length=50,
            ),
            FieldRule(
                field_name="case_name",
                required=False,
                field_type=str,
                max_length=500,
            ),
            FieldRule(
                field_name="plaintiff",
                required=False,
                field_type=str,
                max_length=200,
            ),
            FieldRule(
                field_name="defendant",
                required=False,
                field_type=str,
                max_length=200,
            ),
            FieldRule(
                field_name="case_type",
                required=False,
                field_type=str,
                allowed_values=self.VALID_CASE_TYPES,
            ),
            FieldRule(
                field_name="outcome",
                required=False,
                field_type=str,
                allowed_values=self.VALID_OUTCOMES,
            ),
            FieldRule(
                field_name="filing_date",
                required=False,
                field_type=date,
                min_value=date(1980, 1, 1),
                max_value=date.today(),
            ),
        ]

    def _build_anda_rules(self) -> List[FieldRule]:
        """Build validation rules for ANDA records."""
        return [
            FieldRule(
                field_name="anda_number",
                required=True,
                field_type=str,
                min_length=1,
                max_length=20,
            ),
            FieldRule(
                field_name="generic_company",
                required=True,
                field_type=str,
                min_length=1,
                max_length=200,
            ),
            FieldRule(
                field_name="status",
                required=False,
                field_type=str,
                allowed_values=self.VALID_ANDA_STATUSES,
            ),
        ]

    def _validate_field(
        self,
        value: Any,
        rule: FieldRule,
    ) -> ValidationResult:
        """
        Validate a single field against its rule.

        Args:
            value: The value to validate.
            rule: The validation rule.

        Returns:
            ValidationResult.
        """
        result = ValidationResult(is_valid=True)

        # Check required
        if rule.required and (value is None or value == ""):
            result.add_error(f"Field '{rule.field_name}' is required", rule.field_name)
            return result

        # Skip further validation if value is None and not required
        if value is None:
            return result

        # Check type
        if rule.field_type:
            if rule.field_type == str and not isinstance(value, str):
                result.add_error(
                    f"Field '{rule.field_name}' must be a string",
                    rule.field_name,
                )
            elif rule.field_type == int and not isinstance(value, int):
                result.add_error(
                    f"Field '{rule.field_name}' must be an integer",
                    rule.field_name,
                )
            elif rule.field_type == date and not isinstance(value, (date, datetime)):
                result.add_error(
                    f"Field '{rule.field_name}' must be a date",
                    rule.field_name,
                )

        # String-specific validations
        if isinstance(value, str):
            if rule.min_length and len(value) < rule.min_length:
                result.add_error(
                    f"Field '{rule.field_name}' must be at least {rule.min_length} characters",
                    rule.field_name,
                )
            if rule.max_length and len(value) > rule.max_length:
                result.add_error(
                    f"Field '{rule.field_name}' must be at most {rule.max_length} characters",
                    rule.field_name,
                )
            if rule.pattern and not re.match(rule.pattern, value):
                result.add_error(
                    f"Field '{rule.field_name}' does not match expected pattern",
                    rule.field_name,
                )
            if rule.allowed_values and value.upper() not in rule.allowed_values:
                result.add_error(
                    f"Field '{rule.field_name}' must be one of: {rule.allowed_values}",
                    rule.field_name,
                )

        # Numeric validations
        if isinstance(value, (int, float)):
            if rule.min_value is not None and value < rule.min_value:
                result.add_error(
                    f"Field '{rule.field_name}' must be at least {rule.min_value}",
                    rule.field_name,
                )
            if rule.max_value is not None and value > rule.max_value:
                result.add_error(
                    f"Field '{rule.field_name}' must be at most {rule.max_value}",
                    rule.field_name,
                )

        # Date validations
        if isinstance(value, (date, datetime)):
            if isinstance(value, datetime):
                value = value.date()
            if rule.min_value is not None and isinstance(rule.min_value, date) and value < rule.min_value:
                result.add_error(
                    f"Field '{rule.field_name}' must be on or after {rule.min_value}",
                    rule.field_name,
                )
            if rule.max_value is not None and isinstance(rule.max_value, date) and value > rule.max_value:
                result.add_error(
                    f"Field '{rule.field_name}' must be on or before {rule.max_value}",
                    rule.field_name,
                )

        # Custom validator
        if rule.custom_validator:
            is_valid, error_msg = rule.custom_validator(value)
            if not is_valid:
                result.add_error(error_msg, rule.field_name)

        return result

    def _validate_record(
        self,
        record: Dict[str, Any],
        rules: List[FieldRule],
    ) -> ValidationResult:
        """
        Validate a record against a set of rules.

        Args:
            record: The record to validate.
            rules: List of validation rules.

        Returns:
            ValidationResult.
        """
        result = ValidationResult(is_valid=True)

        for rule in rules:
            value = record.get(rule.field_name)
            field_result = self._validate_field(value, rule)
            result.merge(field_result)

        return result

    def validate_drug(self, record: Dict[str, Any]) -> ValidationResult:
        """
        Validate a drug record.

        Args:
            record: Drug record dictionary.

        Returns:
            ValidationResult.
        """
        result = self._validate_record(record, self._drug_rules)

        # Cross-field validation
        if record.get("annual_revenue") and not record.get("revenue_year"):
            result.add_warning("Annual revenue provided without revenue year")

        return result

    def validate_patent(self, record: Dict[str, Any]) -> ValidationResult:
        """
        Validate a patent record.

        Args:
            record: Patent record dictionary.

        Returns:
            ValidationResult.
        """
        result = self._validate_record(record, self._patent_rules)

        # Cross-field validation: expiration date logic
        base_exp = record.get("base_expiration_date")
        adj_exp = record.get("adjusted_expiration_date")

        if base_exp and adj_exp:
            if isinstance(base_exp, datetime):
                base_exp = base_exp.date()
            if isinstance(adj_exp, datetime):
                adj_exp = adj_exp.date()

            if adj_exp < base_exp:
                result.add_error(
                    "Adjusted expiration date cannot be before base expiration date"
                )

        return result

    def validate_litigation(self, record: Dict[str, Any]) -> ValidationResult:
        """
        Validate a litigation record.

        Args:
            record: Litigation record dictionary.

        Returns:
            ValidationResult.
        """
        result = self._validate_record(record, self._litigation_rules)

        # Cross-field validation
        if record.get("outcome") == "SETTLED" and not record.get("settlement_terms"):
            result.add_warning("Settlement outcome without settlement terms")

        return result

    def validate_anda(self, record: Dict[str, Any]) -> ValidationResult:
        """
        Validate an ANDA record.

        Args:
            record: ANDA record dictionary.

        Returns:
            ValidationResult.
        """
        result = self._validate_record(record, self._anda_rules)

        # Cross-field validation
        if record.get("first_to_file") and not record.get("paragraph_iv_certification"):
            result.add_warning("First-to-file without Paragraph IV certification flag")

        return result

    def validate_batch(
        self,
        records: List[Dict[str, Any]],
        record_type: str,
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[int, ValidationResult]]]:
        """
        Validate a batch of records.

        Args:
            records: List of records to validate.
            record_type: Type of record (drug, patent, litigation, anda).

        Returns:
            Tuple of (valid_records, invalid_records_with_errors).
        """
        validators = {
            "drug": self.validate_drug,
            "patent": self.validate_patent,
            "litigation": self.validate_litigation,
            "anda": self.validate_anda,
        }

        validator = validators.get(record_type)
        if not validator:
            raise ValueError(f"Unknown record type: {record_type}")

        valid_records = []
        invalid_records = []

        for i, record in enumerate(records):
            result = validator(record)
            if result.is_valid:
                valid_records.append(record)
            else:
                invalid_records.append((i, result))

        logger.info(
            f"Batch validation: {len(valid_records)} valid, "
            f"{len(invalid_records)} invalid out of {len(records)} total"
        )

        return valid_records, invalid_records


class DuplicateDetector:
    """
    Detects and handles duplicate records.

    Uses multiple strategies:
    - Exact matching on key fields
    - Fuzzy matching for names
    - Composite key matching
    """

    def __init__(self):
        """Initialize the detector."""
        self._seen_keys: Dict[str, Set[str]] = {}

    def _normalize_string(self, s: str) -> str:
        """Normalize a string for comparison."""
        if not s:
            return ""
        # Lowercase, remove extra whitespace, remove special chars
        s = s.lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s]", "", s)
        return s

    def _generate_drug_key(self, record: Dict[str, Any]) -> str:
        """Generate a unique key for a drug record."""
        parts = []

        # Primary key: NDA number
        if record.get("nda_number"):
            parts.append(f"nda:{record['nda_number']}")

        # Secondary: brand name + company
        if record.get("brand_name"):
            parts.append(f"brand:{self._normalize_string(record['brand_name'])}")
        if record.get("branded_company"):
            parts.append(f"company:{self._normalize_string(record['branded_company'])}")

        return "|".join(parts) if parts else ""

    def _generate_patent_key(self, record: Dict[str, Any]) -> str:
        """Generate a unique key for a patent record."""
        patent_no = record.get("patent_number", "")
        # Normalize patent number
        patent_no = re.sub(r"[,\s]", "", patent_no)
        return f"patent:{patent_no}"

    def _generate_litigation_key(self, record: Dict[str, Any]) -> str:
        """Generate a unique key for a litigation record."""
        parts = []

        if record.get("case_id"):
            parts.append(f"case:{record['case_id']}")
        elif record.get("case_number"):
            parts.append(f"num:{record['case_number']}")

        if record.get("court"):
            parts.append(f"court:{self._normalize_string(record['court'])}")

        return "|".join(parts) if parts else ""

    def _generate_anda_key(self, record: Dict[str, Any]) -> str:
        """Generate a unique key for an ANDA record."""
        return f"anda:{record.get('anda_number', '')}"

    def detect_duplicates(
        self,
        records: List[Dict[str, Any]],
        record_type: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Detect duplicates in a list of records.

        Args:
            records: List of records to check.
            record_type: Type of record (drug, patent, litigation, anda).

        Returns:
            Tuple of (unique_records, duplicate_records).
        """
        key_generators = {
            "drug": self._generate_drug_key,
            "patent": self._generate_patent_key,
            "litigation": self._generate_litigation_key,
            "anda": self._generate_anda_key,
        }

        generator = key_generators.get(record_type)
        if not generator:
            raise ValueError(f"Unknown record type: {record_type}")

        if record_type not in self._seen_keys:
            self._seen_keys[record_type] = set()

        unique_records = []
        duplicate_records = []

        for record in records:
            key = generator(record)

            if not key:
                # Can't generate key, treat as unique
                unique_records.append(record)
                continue

            if key in self._seen_keys[record_type]:
                duplicate_records.append(record)
            else:
                self._seen_keys[record_type].add(key)
                unique_records.append(record)

        logger.info(
            f"Duplicate detection: {len(unique_records)} unique, "
            f"{len(duplicate_records)} duplicates out of {len(records)} total"
        )

        return unique_records, duplicate_records

    def clear(self, record_type: Optional[str] = None) -> None:
        """
        Clear seen keys.

        Args:
            record_type: Type to clear, or None for all.
        """
        if record_type:
            self._seen_keys.pop(record_type, None)
        else:
            self._seen_keys.clear()


class DataQualityReport:
    """Generates data quality reports."""

    def __init__(self):
        """Initialize the reporter."""
        self.metrics: Dict[str, Any] = {}

    def add_validation_results(
        self,
        record_type: str,
        total: int,
        valid: int,
        invalid: int,
        errors: List[str],
    ) -> None:
        """Add validation results to the report."""
        if record_type not in self.metrics:
            self.metrics[record_type] = {}

        self.metrics[record_type]["validation"] = {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "validity_rate": valid / total if total > 0 else 0,
            "top_errors": errors[:10],
        }

    def add_duplicate_results(
        self,
        record_type: str,
        total: int,
        unique: int,
        duplicates: int,
    ) -> None:
        """Add duplicate detection results to the report."""
        if record_type not in self.metrics:
            self.metrics[record_type] = {}

        self.metrics[record_type]["duplicates"] = {
            "total": total,
            "unique": unique,
            "duplicates": duplicates,
            "duplicate_rate": duplicates / total if total > 0 else 0,
        }

    def add_completeness_metrics(
        self,
        record_type: str,
        field_completeness: Dict[str, float],
    ) -> None:
        """Add field completeness metrics."""
        if record_type not in self.metrics:
            self.metrics[record_type] = {}

        self.metrics[record_type]["completeness"] = field_completeness

    def generate_report(self) -> str:
        """Generate a text report."""
        lines = ["=" * 60, "Data Quality Report", "=" * 60, ""]

        for record_type, data in self.metrics.items():
            lines.append(f"\n{record_type.upper()}")
            lines.append("-" * 40)

            if "validation" in data:
                v = data["validation"]
                lines.append(f"  Validation:")
                lines.append(f"    Total records: {v['total']}")
                lines.append(f"    Valid: {v['valid']} ({v['validity_rate']:.1%})")
                lines.append(f"    Invalid: {v['invalid']}")
                if v["top_errors"]:
                    lines.append(f"    Top errors:")
                    for err in v["top_errors"]:
                        lines.append(f"      - {err}")

            if "duplicates" in data:
                d = data["duplicates"]
                lines.append(f"  Duplicates:")
                lines.append(f"    Total: {d['total']}")
                lines.append(f"    Unique: {d['unique']}")
                lines.append(f"    Duplicates: {d['duplicates']} ({d['duplicate_rate']:.1%})")

            if "completeness" in data:
                lines.append(f"  Field Completeness:")
                for field, rate in sorted(data["completeness"].items(), key=lambda x: x[1]):
                    lines.append(f"    {field}: {rate:.1%}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return metrics as dictionary."""
        return self.metrics


# Convenience functions
def validate_required_fields(
    record: Dict[str, Any],
    required_fields: List[str],
) -> ValidationResult:
    """
    Validate that all required fields are present.

    Args:
        record: Record to validate.
        required_fields: List of required field names.

    Returns:
        ValidationResult.
    """
    result = ValidationResult(is_valid=True)

    for field in required_fields:
        if field not in record or record[field] is None or record[field] == "":
            result.add_error(f"Required field '{field}' is missing or empty", field)

    return result


def validate_date_range(
    date_value: Optional[date],
    min_date: Optional[date] = None,
    max_date: Optional[date] = None,
    field_name: str = "date",
) -> ValidationResult:
    """
    Validate a date is within a range.

    Args:
        date_value: Date to validate.
        min_date: Minimum allowed date.
        max_date: Maximum allowed date.
        field_name: Name of the field for error messages.

    Returns:
        ValidationResult.
    """
    result = ValidationResult(is_valid=True)

    if date_value is None:
        return result

    if isinstance(date_value, datetime):
        date_value = date_value.date()

    if min_date and date_value < min_date:
        result.add_error(
            f"Field '{field_name}' ({date_value}) is before minimum ({min_date})",
            field_name,
        )

    if max_date and date_value > max_date:
        result.add_error(
            f"Field '{field_name}' ({date_value}) is after maximum ({max_date})",
            field_name,
        )

    return result


def calculate_field_completeness(
    records: List[Dict[str, Any]],
    fields: List[str],
) -> Dict[str, float]:
    """
    Calculate completeness rate for each field.

    Args:
        records: List of records.
        fields: Fields to check.

    Returns:
        Dictionary of field -> completeness rate.
    """
    if not records:
        return {field: 0.0 for field in fields}

    completeness = {}
    total = len(records)

    for field in fields:
        filled = sum(
            1 for r in records
            if r.get(field) is not None and r.get(field) != ""
        )
        completeness[field] = filled / total

    return completeness


if __name__ == "__main__":
    # Test validation
    validator = DataValidator()

    # Test drug validation
    drug_record = {
        "brand_name": "Humira",
        "generic_name": "adalimumab",
        "nda_number": "125057",
        "branded_company": "AbbVie Inc.",
        "annual_revenue": 21000000000,
        "fda_approval_date": date(2002, 12, 31),
        "market_status": "ACTIVE",
    }

    result = validator.validate_drug(drug_record)
    print(f"Drug validation: {result.is_valid}")
    if not result.is_valid:
        print(f"Errors: {result.errors}")

    # Test patent validation
    patent_record = {
        "patent_number": "6090382",
        "patent_type": "COMPOSITION",
        "base_expiration_date": date(2024, 12, 31),
        "patent_status": "ACTIVE",
    }

    result = validator.validate_patent(patent_record)
    print(f"Patent validation: {result.is_valid}")

    # Test duplicate detection
    detector = DuplicateDetector()

    patents = [
        {"patent_number": "6090382", "patent_type": "COMPOSITION"},
        {"patent_number": "6090382", "patent_type": "METHOD_OF_USE"},  # Duplicate
        {"patent_number": "7223394", "patent_type": "FORMULATION"},
    ]

    unique, duplicates = detector.detect_duplicates(patents, "patent")
    print(f"Unique patents: {len(unique)}, Duplicates: {len(duplicates)}")
