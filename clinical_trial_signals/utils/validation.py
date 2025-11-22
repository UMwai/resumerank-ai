"""
Data validation layer for Clinical Trial Signal Detection System.

Provides validation for all data types used in the system.
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""

    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(f"{field}: {message}" if field else message)


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: Optional[Dict[str, Any]] = None

    def add_error(self, message: str, field: str = None, value: Any = None):
        """Add an error to the result."""
        self.is_valid = False
        self.errors.append(ValidationError(message, field, value))

    def add_warning(self, message: str):
        """Add a warning to the result."""
        self.warnings.append(message)


class BaseValidator:
    """Base class for validators."""

    def validate(self, data: Any) -> ValidationResult:
        """Validate data and return result."""
        raise NotImplementedError


class TrialValidator(BaseValidator):
    """Validator for clinical trial data."""

    # Valid NCT ID pattern
    NCT_PATTERN = re.compile(r'^NCT\d{8}$')

    # Valid trial statuses
    VALID_STATUSES = {
        'NOT_YET_RECRUITING', 'RECRUITING', 'ENROLLING_BY_INVITATION',
        'ACTIVE_NOT_RECRUITING', 'SUSPENDED', 'TERMINATED', 'COMPLETED',
        'WITHDRAWN', 'UNKNOWN'
    }

    # Valid phases
    VALID_PHASES = {
        'EARLY_PHASE1', 'PHASE1', 'PHASE1/PHASE2', 'PHASE2',
        'PHASE2/PHASE3', 'PHASE3', 'PHASE4', 'NA', 'NOT_APPLICABLE'
    }

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate trial data.

        Args:
            data: Dictionary containing trial data

        Returns:
            ValidationResult with validation status and any errors
        """
        result = ValidationResult(is_valid=True)
        sanitized = {}

        # Validate trial_id (required)
        trial_id = data.get('trial_id', '').strip().upper()
        if not trial_id:
            result.add_error("Trial ID is required", "trial_id")
        elif not self.NCT_PATTERN.match(trial_id):
            result.add_error(f"Invalid NCT ID format: {trial_id}", "trial_id", trial_id)
        else:
            sanitized['trial_id'] = trial_id

        # Validate status
        status = data.get('status', '').strip().upper()
        if status:
            if status not in self.VALID_STATUSES:
                result.add_warning(f"Unknown status: {status}")
            sanitized['status'] = status

        # Validate phase
        phase = data.get('phase', '').strip().upper()
        if phase:
            if phase not in self.VALID_PHASES:
                result.add_warning(f"Unknown phase: {phase}")
            sanitized['phase'] = phase

        # Validate enrollment
        enrollment_target = data.get('enrollment_target')
        if enrollment_target is not None:
            try:
                enrollment_target = int(enrollment_target)
                if enrollment_target < 0:
                    result.add_error("Enrollment target cannot be negative", "enrollment_target", enrollment_target)
                elif enrollment_target > 1000000:
                    result.add_warning(f"Unusually large enrollment target: {enrollment_target}")
                sanitized['enrollment_target'] = enrollment_target
            except (TypeError, ValueError):
                result.add_error("Invalid enrollment target", "enrollment_target", enrollment_target)

        # Validate dates
        for date_field in ['start_date', 'expected_completion', 'primary_completion_date']:
            date_value = data.get(date_field)
            validated_date = self._validate_date(date_value)
            if validated_date is not None:
                sanitized[date_field] = validated_date

        # Validate sponsor
        sponsor = data.get('sponsor', '').strip()
        if sponsor:
            sanitized['sponsor'] = sponsor[:300]  # Truncate to max length

        # Copy other fields
        for field in ['drug_name', 'indication', 'primary_endpoint', 'study_type', 'company_ticker']:
            if field in data and data[field]:
                sanitized[field] = str(data[field]).strip()

        result.sanitized_data = sanitized
        return result

    def _validate_date(self, value: Any) -> Optional[date]:
        """Validate and parse a date value."""
        if value is None:
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            try:
                # Try ISO format first
                if len(value) >= 10:
                    return datetime.strptime(value[:10], '%Y-%m-%d').date()
                elif len(value) == 7:  # YYYY-MM format
                    return datetime.strptime(value, '%Y-%m').date()
            except ValueError:
                pass
        return None


class SignalValidator(BaseValidator):
    """Validator for trial signals."""

    # Valid signal types
    VALID_SIGNAL_TYPES = {
        # Positive signals
        'sites_added', 'insider_buying', 'early_enrollment', 'patent_filed',
        'late_breaking_abstract', 'ceo_presentation', 'status_change_positive',
        'completion_date_accelerated', 'enrollment_increase', 'sec_8k_positive',
        'preprint_positive', 'fda_breakthrough', 'fda_fast_track', 'fda_priority_review',
        # Negative signals
        'enrollment_extended', 'endpoint_change', 'insider_selling', 'sites_removed',
        'no_conference', 'risk_factor_increase', 'status_change_negative',
        'completion_date_delayed', 'enrollment_decrease', 'sec_8k_negative',
        'preprint_negative', 'clinical_hold', 'safety_signal',
        # Neutral signals
        'status_change_neutral', 'data_update',
    }

    VALID_SOURCES = {
        'clinicaltrials', 'sec_edgar', 'pubmed', 'medrxiv', 'biorxiv',
        'uspto', 'change_detection', 'manual', 'fda'
    }

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate signal data.

        Args:
            data: Dictionary containing signal data

        Returns:
            ValidationResult with validation status and any errors
        """
        result = ValidationResult(is_valid=True)
        sanitized = {}

        # Validate trial_id (required)
        trial_id = data.get('trial_id', '').strip().upper()
        if not trial_id:
            result.add_error("Trial ID is required", "trial_id")
        else:
            sanitized['trial_id'] = trial_id

        # Validate signal_type (required)
        signal_type = data.get('signal_type', '').strip().lower()
        if not signal_type:
            result.add_error("Signal type is required", "signal_type")
        else:
            if signal_type not in self.VALID_SIGNAL_TYPES:
                result.add_warning(f"Unknown signal type: {signal_type}")
            sanitized['signal_type'] = signal_type

        # Validate signal_weight
        signal_weight = data.get('signal_weight', 0)
        try:
            signal_weight = int(signal_weight)
            if signal_weight < -5 or signal_weight > 5:
                result.add_error("Signal weight must be between -5 and 5", "signal_weight", signal_weight)
            else:
                sanitized['signal_weight'] = signal_weight
        except (TypeError, ValueError):
            result.add_error("Invalid signal weight", "signal_weight", signal_weight)

        # Validate source
        source = data.get('source', '').strip().lower()
        if source:
            if source not in self.VALID_SOURCES:
                result.add_warning(f"Unknown source: {source}")
            sanitized['source'] = source

        # Copy other fields
        if 'signal_value' in data:
            sanitized['signal_value'] = str(data['signal_value'])[:1000]
        if 'source_url' in data:
            sanitized['source_url'] = str(data['source_url'])[:2000]

        result.sanitized_data = sanitized
        return result


class CompanyValidator(BaseValidator):
    """Validator for company data."""

    # Valid ticker pattern (1-5 uppercase letters)
    TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}$')

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate company data.

        Args:
            data: Dictionary containing company data

        Returns:
            ValidationResult with validation status and any errors
        """
        result = ValidationResult(is_valid=True)
        sanitized = {}

        # Validate ticker (required)
        ticker = data.get('ticker', '').strip().upper()
        if not ticker:
            result.add_error("Ticker is required", "ticker")
        elif not self.TICKER_PATTERN.match(ticker):
            result.add_error(f"Invalid ticker format: {ticker}", "ticker", ticker)
        else:
            sanitized['ticker'] = ticker

        # Validate company_name (required)
        company_name = data.get('company_name', '').strip()
        if not company_name:
            result.add_error("Company name is required", "company_name")
        else:
            sanitized['company_name'] = company_name[:200]

        # Validate market_cap
        market_cap = data.get('market_cap')
        if market_cap is not None:
            try:
                market_cap = int(market_cap)
                if market_cap < 0:
                    result.add_error("Market cap cannot be negative", "market_cap", market_cap)
                else:
                    sanitized['market_cap'] = market_cap
            except (TypeError, ValueError):
                result.add_error("Invalid market cap", "market_cap", market_cap)

        # Validate CIK
        cik = data.get('cik', '').strip()
        if cik:
            if not cik.isdigit():
                result.add_warning(f"CIK should be numeric: {cik}")
            sanitized['cik'] = cik.zfill(10)[:20]

        result.sanitized_data = sanitized
        return result


class PreprintValidator(BaseValidator):
    """Validator for preprint/publication data."""

    # PMID pattern (numeric)
    PMID_PATTERN = re.compile(r'^\d+$')

    # DOI pattern
    DOI_PATTERN = re.compile(r'^10\.\d{4,}/[\w\.\-/]+$')

    VALID_SOURCES = {'pubmed', 'medrxiv', 'biorxiv'}

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate preprint data.

        Args:
            data: Dictionary containing preprint data

        Returns:
            ValidationResult with validation status and any errors
        """
        result = ValidationResult(is_valid=True)
        sanitized = {}

        # Validate source
        source = data.get('source', '').strip().lower()
        if not source:
            result.add_error("Source is required", "source")
        elif source not in self.VALID_SOURCES:
            result.add_error(f"Invalid source: {source}", "source", source)
        else:
            sanitized['source'] = source

        # Validate external_id (PMID or DOI)
        external_id = data.get('external_id', '').strip()
        if not external_id:
            result.add_error("External ID is required", "external_id")
        else:
            if source == 'pubmed':
                if not self.PMID_PATTERN.match(external_id):
                    result.add_warning(f"Unusual PMID format: {external_id}")
            elif source in ('medrxiv', 'biorxiv'):
                if not self.DOI_PATTERN.match(external_id):
                    result.add_warning(f"Unusual DOI format: {external_id}")
            sanitized['external_id'] = external_id

        # Validate title
        title = data.get('title', '').strip()
        if title:
            sanitized['title'] = title[:2000]
        else:
            result.add_warning("Title is missing")

        # Validate publication_date
        pub_date = data.get('publication_date')
        if pub_date:
            if isinstance(pub_date, date):
                sanitized['publication_date'] = pub_date
            elif isinstance(pub_date, str):
                try:
                    sanitized['publication_date'] = datetime.strptime(pub_date[:10], '%Y-%m-%d').date()
                except ValueError:
                    result.add_warning(f"Invalid publication date: {pub_date}")

        result.sanitized_data = sanitized
        return result


class PatentValidator(BaseValidator):
    """Validator for patent data."""

    # US patent number pattern
    US_PATENT_PATTERN = re.compile(r'^(US)?[\d,]+$')

    # US application number pattern
    US_APP_PATTERN = re.compile(r'^\d{2}/\d{3},?\d{3}$')

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate patent data.

        Args:
            data: Dictionary containing patent data

        Returns:
            ValidationResult with validation status and any errors
        """
        result = ValidationResult(is_valid=True)
        sanitized = {}

        # Validate patent_number or application_number (at least one required)
        patent_number = data.get('patent_number', '').strip().upper()
        application_number = data.get('application_number', '').strip()

        if not patent_number and not application_number:
            result.add_error("Either patent_number or application_number is required")

        if patent_number:
            # Clean up patent number
            patent_clean = re.sub(r'[,\s]', '', patent_number)
            sanitized['patent_number'] = patent_clean[:50]

        if application_number:
            sanitized['application_number'] = application_number[:50]

        # Validate title
        title = data.get('title', '').strip()
        if title:
            sanitized['title'] = title[:2000]

        # Validate assignee
        assignee = data.get('assignee', '').strip()
        if assignee:
            sanitized['assignee'] = assignee[:300]

        # Validate dates
        for date_field in ['filing_date', 'publication_date', 'grant_date']:
            date_value = data.get(date_field)
            if date_value:
                if isinstance(date_value, date):
                    sanitized[date_field] = date_value
                elif isinstance(date_value, str):
                    try:
                        sanitized[date_field] = datetime.strptime(date_value[:10], '%Y-%m-%d').date()
                    except ValueError:
                        result.add_warning(f"Invalid {date_field}: {date_value}")

        result.sanitized_data = sanitized
        return result


# Convenience functions
def validate_trial(data: Dict[str, Any]) -> ValidationResult:
    """Validate trial data."""
    return TrialValidator().validate(data)


def validate_signal(data: Dict[str, Any]) -> ValidationResult:
    """Validate signal data."""
    return SignalValidator().validate(data)


def validate_company(data: Dict[str, Any]) -> ValidationResult:
    """Validate company data."""
    return CompanyValidator().validate(data)


def validate_preprint(data: Dict[str, Any]) -> ValidationResult:
    """Validate preprint data."""
    return PreprintValidator().validate(data)


def validate_patent(data: Dict[str, Any]) -> ValidationResult:
    """Validate patent data."""
    return PatentValidator().validate(data)
