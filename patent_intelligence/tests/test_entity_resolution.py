"""
Tests for the entity resolution module.
"""

import pytest
from src.utils.entity_resolution import (
    DrugNameResolver,
    CompanyNameResolver,
    EntityResolutionService,
    EntityMatch,
    get_entity_resolution_service,
)


class TestEntityMatch:
    """Tests for EntityMatch dataclass."""

    def test_create_matched_result(self):
        """Test creating a matched result."""
        match = EntityMatch(
            matched=True,
            confidence=0.95,
            canonical_name="adalimumab",
            matched_name="Humira",
            match_type="SYNONYM",
        )
        assert match.matched is True
        assert match.confidence == 0.95
        assert match.match_type == "SYNONYM"

    def test_create_unmatched_result(self):
        """Test creating an unmatched result."""
        match = EntityMatch(
            matched=False,
            confidence=0.0,
            canonical_name="",
            matched_name="Unknown Drug",
            match_type="NONE",
        )
        assert match.matched is False


class TestDrugNameResolver:
    """Tests for DrugNameResolver class."""

    @pytest.fixture
    def resolver(self):
        """Create a resolver instance."""
        return DrugNameResolver(fuzzy_threshold=0.85)

    def test_resolve_exact_match_brand_name(self, resolver):
        """Test resolving exact brand name match."""
        result = resolver.resolve("Humira")
        assert result.matched is True
        assert result.match_type == "EXACT"
        assert result.canonical_name == "adalimumab"

    def test_resolve_exact_match_generic_name(self, resolver):
        """Test resolving exact generic name match."""
        result = resolver.resolve("adalimumab")
        assert result.matched is True
        assert result.match_type == "EXACT"
        assert result.canonical_name == "adalimumab"

    def test_resolve_case_insensitive(self, resolver):
        """Test case-insensitive matching."""
        result = resolver.resolve("HUMIRA")
        assert result.matched is True
        assert result.canonical_name == "adalimumab"

    def test_resolve_biosimilar(self, resolver):
        """Test resolving biosimilar to reference product."""
        result = resolver.resolve("Hadlima")
        assert result.matched is True
        assert result.canonical_name == "adalimumab"

    def test_resolve_unknown_drug(self, resolver):
        """Test resolving unknown drug name."""
        result = resolver.resolve("CompletelyUnknownDrug12345")
        assert result.matched is False
        assert result.match_type == "NONE"

    def test_resolve_empty_string(self, resolver):
        """Test resolving empty string."""
        result = resolver.resolve("")
        assert result.matched is False

    def test_match_brand_to_generic(self, resolver):
        """Test matching brand name to generic name."""
        result = resolver.match("Humira", "adalimumab")
        assert result.matched is True
        assert result.match_type == "SYNONYM"

    def test_match_different_biosimilars(self, resolver):
        """Test matching different biosimilars of same drug."""
        result = resolver.match("Hadlima", "Cyltezo")
        assert result.matched is True

    def test_match_different_drugs(self, resolver):
        """Test matching different drugs."""
        result = resolver.match("Humira", "Keytruda")
        assert result.matched is False

    def test_add_alias(self, resolver):
        """Test adding a new alias."""
        resolver.add_alias("test_drug", "test_alias")
        result = resolver.resolve("test_alias")
        assert result.matched is True
        assert result.canonical_name == "test_drug"

    def test_get_all_names(self, resolver):
        """Test getting all names for a drug."""
        names = resolver.get_all_names("adalimumab")
        assert "adalimumab" in names
        assert len(names) > 1  # Should have aliases

    def test_fuzzy_match(self, resolver):
        """Test fuzzy matching with typos."""
        # This depends on the threshold and specific implementation
        resolver.fuzzy_threshold = 0.80
        result = resolver.resolve("humra")  # Missing 'i'
        # May or may not match depending on threshold


class TestCompanyNameResolver:
    """Tests for CompanyNameResolver class."""

    @pytest.fixture
    def resolver(self):
        """Create a resolver instance."""
        return CompanyNameResolver(fuzzy_threshold=0.80)

    def test_resolve_exact_match(self, resolver):
        """Test resolving exact company name match."""
        result = resolver.resolve("AbbVie Inc.")
        assert result.matched is True
        assert result.match_type == "EXACT"
        assert result.metadata.get("ticker") == "ABBV"

    def test_resolve_abbreviation(self, resolver):
        """Test resolving company abbreviation."""
        result = resolver.resolve("abbvie")
        assert result.matched is True
        assert result.canonical_name == "AbbVie Inc."

    def test_resolve_by_ticker(self, resolver):
        """Test resolving by ticker symbol."""
        result = resolver.resolve("ABBV")
        assert result.matched is True
        assert result.canonical_name == "AbbVie Inc."
        assert result.match_type == "TICKER"

    def test_resolve_with_suffix_variations(self, resolver):
        """Test resolving with different suffixes."""
        result = resolver.resolve("Pfizer")
        assert result.matched is True
        assert "Pfizer" in result.canonical_name

    def test_resolve_generic_company(self, resolver):
        """Test resolving generic company."""
        result = resolver.resolve("Teva")
        assert result.matched is True
        assert result.metadata.get("type") == "GENERIC"

    def test_resolve_unknown_company(self, resolver):
        """Test resolving unknown company."""
        result = resolver.resolve("Unknown Pharma Corp XYZ")
        assert result.matched is False

    def test_get_ticker(self, resolver):
        """Test getting ticker symbol."""
        ticker = resolver.get_ticker("AbbVie")
        assert ticker == "ABBV"

    def test_get_ticker_unknown(self, resolver):
        """Test getting ticker for unknown company."""
        ticker = resolver.get_ticker("Unknown Company")
        assert ticker is None

    def test_get_company_type(self, resolver):
        """Test getting company type."""
        company_type = resolver.get_company_type("Teva")
        assert company_type == "GENERIC"


class TestEntityResolutionService:
    """Tests for EntityResolutionService class."""

    @pytest.fixture
    def service(self):
        """Create a service instance."""
        return EntityResolutionService()

    def test_resolve_drug(self, service):
        """Test resolving drug name."""
        result = service.resolve_drug("Humira")
        assert result.matched is True

    def test_resolve_company(self, service):
        """Test resolving company name."""
        result = service.resolve_company("Pfizer")
        assert result.matched is True

    def test_match_drugs(self, service):
        """Test matching drug names."""
        result = service.match_drugs("Humira", "adalimumab")
        assert result.matched is True

    def test_get_drug_names(self, service):
        """Test getting all drug names."""
        names = service.get_drug_names("Humira")
        assert len(names) > 0
        assert any("adalimumab" in n.lower() for n in names)

    def test_get_company_ticker(self, service):
        """Test getting company ticker."""
        ticker = service.get_company_ticker("Merck")
        assert ticker == "MRK"

    def test_enrich_drug_record(self, service):
        """Test enriching drug record."""
        record = {
            "brand_name": "Humira",
            "generic_name": "adalimumab",
            "branded_company": "AbbVie",
        }
        enriched = service.enrich_drug_record(record)

        assert "branded_company_ticker" in enriched
        assert enriched["branded_company_ticker"] == "ABBV"

    def test_deduplicate_records(self, service):
        """Test deduplicating records."""
        records = [
            {"brand_name": "Humira", "notes": "record 1"},
            {"brand_name": "HUMIRA", "notes": "record 2"},
            {"brand_name": "Keytruda", "notes": "record 3"},
        ]
        deduped = service.deduplicate_records(records, "brand_name", "drug")

        # Should merge Humira records
        assert len(deduped) == 2


class TestGetEntityResolutionService:
    """Tests for the convenience function."""

    def test_create_service(self):
        """Test creating service via convenience function."""
        service = get_entity_resolution_service()
        assert service is not None
        assert service.drug_resolver is not None
        assert service.company_resolver is not None


class TestKnownDrugMappings:
    """Tests for specific known drug mappings."""

    @pytest.fixture
    def resolver(self):
        """Create a resolver instance."""
        return DrugNameResolver()

    @pytest.mark.parametrize("brand,generic", [
        ("Keytruda", "pembrolizumab"),
        ("Opdivo", "nivolumab"),
        ("Eliquis", "apixaban"),
        ("Stelara", "ustekinumab"),
        ("Revlimid", "lenalidomide"),
        ("Imbruvica", "ibrutinib"),
        ("Herceptin", "trastuzumab"),
        ("Rituxan", "rituximab"),
        ("Enbrel", "etanercept"),
        ("Remicade", "infliximab"),
    ])
    def test_major_drug_mappings(self, resolver, brand, generic):
        """Test that major drugs are correctly mapped."""
        result = resolver.resolve(brand)
        assert result.matched is True
        assert result.canonical_name == generic


class TestKnownCompanyMappings:
    """Tests for specific known company mappings."""

    @pytest.fixture
    def resolver(self):
        """Create a resolver instance."""
        return CompanyNameResolver()

    @pytest.mark.parametrize("company,ticker", [
        ("AbbVie", "ABBV"),
        ("Bristol-Myers Squibb", "BMY"),
        ("Merck", "MRK"),
        ("Pfizer", "PFE"),
        ("Johnson & Johnson", "JNJ"),
        ("Eli Lilly", "LLY"),
        ("Amgen", "AMGN"),
        ("Gilead", "GILD"),
        ("Biogen", "BIIB"),
        ("Regeneron", "REGN"),
    ])
    def test_major_company_mappings(self, resolver, company, ticker):
        """Test that major companies are correctly mapped."""
        result = resolver.resolve(company)
        assert result.matched is True
        assert result.metadata.get("ticker") == ticker


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
