"""
Entity Resolution Module

Provides entity resolution and matching capabilities for drug names,
company names, and other entities across different data sources.

Uses multiple strategies:
- Exact matching
- Fuzzy string matching
- Synonym/alias lookup
- Active ingredient matching
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class EntityMatch:
    """Result of an entity matching operation."""
    matched: bool
    confidence: float  # 0.0 to 1.0
    canonical_name: str
    matched_name: str
    match_type: str  # EXACT, FUZZY, SYNONYM, INGREDIENT
    metadata: Dict[str, Any] = field(default_factory=dict)


class DrugNameResolver:
    """
    Resolves and matches drug names across different sources.

    Handles:
    - Brand name variations (e.g., "Humira" vs "HUMIRA")
    - Generic name matching (e.g., "adalimumab")
    - Active ingredient matching
    - Common misspellings and abbreviations
    """

    # Known drug name synonyms and aliases
    # Format: canonical_name -> [aliases]
    DRUG_ALIASES: Dict[str, List[str]] = {
        "adalimumab": ["humira", "hadlima", "hyrimoz", "cyltezo", "amjevita", "imraldi"],
        "pembrolizumab": ["keytruda"],
        "nivolumab": ["opdivo"],
        "apixaban": ["eliquis"],
        "rivaroxaban": ["xarelto"],
        "ustekinumab": ["stelara"],
        "lenalidomide": ["revlimid"],
        "ibrutinib": ["imbruvica"],
        "trastuzumab": ["herceptin", "herzuma", "ogivri", "ontruzant", "kanjinti"],
        "rituximab": ["rituxan", "truxima", "ruxience"],
        "bevacizumab": ["avastin", "mvasi", "zirabev"],
        "infliximab": ["remicade", "inflectra", "renflexis", "ixifi"],
        "etanercept": ["enbrel", "erelzi", "eticovo"],
        "denosumab": ["prolia", "xgeva"],
        "golimumab": ["simponi"],
        "secukinumab": ["cosentyx"],
        "ixekizumab": ["taltz"],
        "dupilumab": ["dupixent"],
        "guselkumab": ["tremfya"],
        "risankizumab": ["skyrizi"],
        "tildrakizumab": ["ilumya"],
        "certolizumab pegol": ["cimzia"],
        "vedolizumab": ["entyvio"],
        "natalizumab": ["tysabri"],
        "tocilizumab": ["actemra"],
        "sarilumab": ["kevzara"],
        "baricitinib": ["olumiant"],
        "tofacitinib": ["xeljanz"],
        "upadacitinib": ["rinvoq"],
        "abatacept": ["orencia"],
        "belimumab": ["benlysta"],
        "daratumumab": ["darzalex"],
        "elotuzumab": ["empliciti"],
        "bortezomib": ["velcade"],
        "carfilzomib": ["kyprolis"],
        "ixazomib": ["ninlaro"],
        "pomalidomide": ["pomalyst"],
        "thalidomide": ["thalomid"],
        "venetoclax": ["venclexta"],
        "imatinib": ["gleevec"],
        "dasatinib": ["sprycel"],
        "nilotinib": ["tasigna"],
        "bosutinib": ["bosulif"],
        "ponatinib": ["iclusig"],
        "sunitinib": ["sutent"],
        "sorafenib": ["nexavar"],
        "regorafenib": ["stivarga"],
        "cabozantinib": ["cabometyx", "cometriq"],
        "axitinib": ["inlyta"],
        "pazopanib": ["votrient"],
        "lenvatinib": ["lenvima"],
        "erlotinib": ["tarceva"],
        "gefitinib": ["iressa"],
        "afatinib": ["gilotrif"],
        "osimertinib": ["tagrisso"],
        "crizotinib": ["xalkori"],
        "ceritinib": ["zykadia"],
        "alectinib": ["alecensa"],
        "brigatinib": ["alunbrig"],
        "lorlatinib": ["lorbrena"],
        "vemurafenib": ["zelboraf"],
        "dabrafenib": ["tafinlar"],
        "encorafenib": ["braftovi"],
        "trametinib": ["mekinist"],
        "cobimetinib": ["cotellic"],
        "binimetinib": ["mektovi"],
        "palbociclib": ["ibrance"],
        "ribociclib": ["kisqali"],
        "abemaciclib": ["verzenio"],
        "olaparib": ["lynparza"],
        "niraparib": ["zejula"],
        "rucaparib": ["rubraca"],
        "talazoparib": ["talzenna"],
        "abiraterone": ["zytiga"],
        "enzalutamide": ["xtandi"],
        "apalutamide": ["erleada"],
        "darolutamide": ["nubeqa"],
        "sipuleucel-t": ["provenge"],
        "atezolizumab": ["tecentriq"],
        "durvalumab": ["imfinzi"],
        "avelumab": ["bavencio"],
        "cemiplimab": ["libtayo"],
        "ipilimumab": ["yervoy"],
        "tremelimumab": ["imjudo"],
    }

    # Build reverse mapping for faster lookup
    _alias_to_canonical: Dict[str, str] = {}

    def __init__(self, fuzzy_threshold: float = 0.85):
        """
        Initialize the drug name resolver.

        Args:
            fuzzy_threshold: Minimum similarity score for fuzzy matching.
        """
        self.fuzzy_threshold = fuzzy_threshold

        # Build reverse alias mapping
        self._build_alias_mapping()

        # Cache for resolved names
        self._cache: Dict[str, EntityMatch] = {}

    def _build_alias_mapping(self) -> None:
        """Build reverse mapping from alias to canonical name."""
        for canonical, aliases in self.DRUG_ALIASES.items():
            self._alias_to_canonical[self._normalize(canonical)] = canonical
            for alias in aliases:
                self._alias_to_canonical[self._normalize(alias)] = canonical

    def _normalize(self, name: str) -> str:
        """Normalize a drug name for comparison."""
        if not name:
            return ""
        # Lowercase, remove special characters, normalize whitespace
        name = name.lower().strip()
        name = re.sub(r"[^\w\s]", "", name)
        name = re.sub(r"\s+", " ", name)
        return name

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using SequenceMatcher."""
        return SequenceMatcher(None, s1, s2).ratio()

    def resolve(self, name: str) -> EntityMatch:
        """
        Resolve a drug name to its canonical form.

        Args:
            name: Drug name to resolve.

        Returns:
            EntityMatch with resolution results.
        """
        if not name:
            return EntityMatch(
                matched=False,
                confidence=0.0,
                canonical_name="",
                matched_name=name,
                match_type="NONE",
            )

        normalized = self._normalize(name)

        # Check cache
        if normalized in self._cache:
            return self._cache[normalized]

        # Try exact match first
        if normalized in self._alias_to_canonical:
            result = EntityMatch(
                matched=True,
                confidence=1.0,
                canonical_name=self._alias_to_canonical[normalized],
                matched_name=name,
                match_type="EXACT",
            )
            self._cache[normalized] = result
            return result

        # Try fuzzy matching
        best_match = None
        best_score = 0.0

        for alias, canonical in self._alias_to_canonical.items():
            score = self._calculate_similarity(normalized, alias)
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = canonical

        if best_match:
            result = EntityMatch(
                matched=True,
                confidence=best_score,
                canonical_name=best_match,
                matched_name=name,
                match_type="FUZZY",
            )
            self._cache[normalized] = result
            return result

        # No match found
        result = EntityMatch(
            matched=False,
            confidence=0.0,
            canonical_name=name,  # Use original as canonical
            matched_name=name,
            match_type="NONE",
        )
        self._cache[normalized] = result
        return result

    def match(self, name1: str, name2: str) -> EntityMatch:
        """
        Check if two drug names refer to the same drug.

        Args:
            name1: First drug name.
            name2: Second drug name.

        Returns:
            EntityMatch indicating if names match.
        """
        resolution1 = self.resolve(name1)
        resolution2 = self.resolve(name2)

        # Both resolved to same canonical name
        if resolution1.matched and resolution2.matched:
            if resolution1.canonical_name == resolution2.canonical_name:
                return EntityMatch(
                    matched=True,
                    confidence=min(resolution1.confidence, resolution2.confidence),
                    canonical_name=resolution1.canonical_name,
                    matched_name=f"{name1} <-> {name2}",
                    match_type="SYNONYM",
                )

        # Check direct fuzzy match between original names
        norm1 = self._normalize(name1)
        norm2 = self._normalize(name2)
        score = self._calculate_similarity(norm1, norm2)

        if score >= self.fuzzy_threshold:
            return EntityMatch(
                matched=True,
                confidence=score,
                canonical_name=name1,  # Use first as canonical
                matched_name=f"{name1} <-> {name2}",
                match_type="FUZZY",
            )

        return EntityMatch(
            matched=False,
            confidence=score,
            canonical_name="",
            matched_name=f"{name1} <-> {name2}",
            match_type="NONE",
        )

    def add_alias(self, canonical: str, alias: str) -> None:
        """
        Add a new alias for a drug.

        Args:
            canonical: Canonical drug name.
            alias: Alias to add.
        """
        norm_canonical = self._normalize(canonical)
        norm_alias = self._normalize(alias)

        if norm_canonical not in self.DRUG_ALIASES:
            self.DRUG_ALIASES[norm_canonical] = []
        if alias not in self.DRUG_ALIASES[norm_canonical]:
            self.DRUG_ALIASES[norm_canonical].append(alias)

        self._alias_to_canonical[norm_alias] = norm_canonical

        # Clear cache
        self._cache.clear()

    def get_all_names(self, canonical: str) -> List[str]:
        """
        Get all known names for a drug.

        Args:
            canonical: Canonical drug name.

        Returns:
            List of all known names including the canonical name.
        """
        norm = self._normalize(canonical)
        names = [canonical]

        if norm in self.DRUG_ALIASES:
            names.extend(self.DRUG_ALIASES[norm])

        return names


class CompanyNameResolver:
    """
    Resolves and matches pharmaceutical company names.

    Handles:
    - Full company names vs abbreviations
    - Name variations (Inc., Corp., Ltd., etc.)
    - Parent company / subsidiary relationships
    - Ticker symbol mapping
    """

    # Company name aliases and ticker mapping
    COMPANY_DATA: Dict[str, Dict[str, Any]] = {
        "AbbVie Inc.": {
            "aliases": ["abbvie", "abbv"],
            "ticker": "ABBV",
            "type": "BRANDED",
        },
        "Bristol-Myers Squibb": {
            "aliases": ["bms", "bristol myers", "bristol-myers"],
            "ticker": "BMY",
            "type": "BRANDED",
        },
        "Merck & Co.": {
            "aliases": ["merck", "msd", "merck sharp dohme", "merck sharp & dohme"],
            "ticker": "MRK",
            "type": "BRANDED",
        },
        "Pfizer Inc.": {
            "aliases": ["pfizer", "pfe"],
            "ticker": "PFE",
            "type": "BOTH",
        },
        "Johnson & Johnson": {
            "aliases": ["j&j", "jnj", "janssen"],
            "ticker": "JNJ",
            "type": "BOTH",
        },
        "Eli Lilly": {
            "aliases": ["lilly", "eli lilly and company"],
            "ticker": "LLY",
            "type": "BRANDED",
        },
        "Novartis": {
            "aliases": ["novartis ag", "novartis pharma"],
            "ticker": "NVS",
            "type": "BOTH",
        },
        "Roche": {
            "aliases": ["roche holding", "genentech", "roche pharma"],
            "ticker": "RHHBY",
            "type": "BRANDED",
        },
        "AstraZeneca": {
            "aliases": ["az", "astrazeneca plc"],
            "ticker": "AZN",
            "type": "BRANDED",
        },
        "Sanofi": {
            "aliases": ["sanofi-aventis", "sanofi sa"],
            "ticker": "SNY",
            "type": "BRANDED",
        },
        "GlaxoSmithKline": {
            "aliases": ["gsk", "glaxo"],
            "ticker": "GSK",
            "type": "BRANDED",
        },
        "Amgen Inc.": {
            "aliases": ["amgen"],
            "ticker": "AMGN",
            "type": "BRANDED",
        },
        "Gilead Sciences": {
            "aliases": ["gilead"],
            "ticker": "GILD",
            "type": "BRANDED",
        },
        "Biogen": {
            "aliases": ["biogen idec"],
            "ticker": "BIIB",
            "type": "BRANDED",
        },
        "Regeneron": {
            "aliases": ["regeneron pharmaceuticals"],
            "ticker": "REGN",
            "type": "BRANDED",
        },
        "Vertex Pharmaceuticals": {
            "aliases": ["vertex"],
            "ticker": "VRTX",
            "type": "BRANDED",
        },
        "Teva Pharmaceutical": {
            "aliases": ["teva", "teva pharma"],
            "ticker": "TEVA",
            "type": "GENERIC",
        },
        "Viatris": {
            "aliases": ["mylan", "mylan labs", "upjohn"],
            "ticker": "VTRS",
            "type": "GENERIC",
        },
        "Sandoz": {
            "aliases": ["sandoz inc"],
            "ticker": None,
            "type": "GENERIC",
        },
        "Sun Pharmaceutical": {
            "aliases": ["sun pharma"],
            "ticker": "SUNPHARMA.NS",
            "type": "GENERIC",
        },
        "Dr. Reddy's": {
            "aliases": ["dr reddys", "dr reddy"],
            "ticker": "RDY",
            "type": "GENERIC",
        },
        "Cipla": {
            "aliases": ["cipla ltd"],
            "ticker": "CIPLA.NS",
            "type": "GENERIC",
        },
        "Lupin": {
            "aliases": ["lupin limited", "lupin pharma"],
            "ticker": "LUPIN.NS",
            "type": "GENERIC",
        },
        "Aurobindo Pharma": {
            "aliases": ["aurobindo"],
            "ticker": "AUROPHARMA.NS",
            "type": "GENERIC",
        },
        "Hikma Pharmaceuticals": {
            "aliases": ["hikma"],
            "ticker": "HIK.L",
            "type": "GENERIC",
        },
        "Fresenius Kabi": {
            "aliases": ["fresenius"],
            "ticker": "FRE.DE",
            "type": "GENERIC",
        },
    }

    # Build lookup tables
    _alias_to_canonical: Dict[str, str] = {}
    _ticker_to_canonical: Dict[str, str] = {}

    def __init__(self, fuzzy_threshold: float = 0.80):
        """
        Initialize the company name resolver.

        Args:
            fuzzy_threshold: Minimum similarity score for fuzzy matching.
        """
        self.fuzzy_threshold = fuzzy_threshold
        self._build_mappings()
        self._cache: Dict[str, EntityMatch] = {}

    def _build_mappings(self) -> None:
        """Build lookup mappings from company data."""
        for canonical, data in self.COMPANY_DATA.items():
            norm_canonical = self._normalize(canonical)
            self._alias_to_canonical[norm_canonical] = canonical

            for alias in data.get("aliases", []):
                self._alias_to_canonical[self._normalize(alias)] = canonical

            if data.get("ticker"):
                self._ticker_to_canonical[data["ticker"].upper()] = canonical

    def _normalize(self, name: str) -> str:
        """Normalize a company name for comparison."""
        if not name:
            return ""
        name = name.lower().strip()
        # Remove common suffixes
        suffixes = [" inc", " inc.", " corp", " corp.", " ltd", " ltd.",
                    " llc", " plc", " ag", " sa", " nv", " se"]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        # Remove special characters
        name = re.sub(r"[^\w\s]", "", name)
        name = re.sub(r"\s+", " ", name)
        return name.strip()

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity."""
        return SequenceMatcher(None, s1, s2).ratio()

    def resolve(self, name: str) -> EntityMatch:
        """
        Resolve a company name to its canonical form.

        Args:
            name: Company name to resolve.

        Returns:
            EntityMatch with resolution results.
        """
        if not name:
            return EntityMatch(
                matched=False,
                confidence=0.0,
                canonical_name="",
                matched_name=name,
                match_type="NONE",
            )

        normalized = self._normalize(name)

        # Check cache
        if normalized in self._cache:
            return self._cache[normalized]

        # Try exact match
        if normalized in self._alias_to_canonical:
            canonical = self._alias_to_canonical[normalized]
            result = EntityMatch(
                matched=True,
                confidence=1.0,
                canonical_name=canonical,
                matched_name=name,
                match_type="EXACT",
                metadata={
                    "ticker": self.COMPANY_DATA[canonical].get("ticker"),
                    "type": self.COMPANY_DATA[canonical].get("type"),
                },
            )
            self._cache[normalized] = result
            return result

        # Try ticker match
        if name.upper() in self._ticker_to_canonical:
            canonical = self._ticker_to_canonical[name.upper()]
            result = EntityMatch(
                matched=True,
                confidence=1.0,
                canonical_name=canonical,
                matched_name=name,
                match_type="TICKER",
                metadata={
                    "ticker": self.COMPANY_DATA[canonical].get("ticker"),
                    "type": self.COMPANY_DATA[canonical].get("type"),
                },
            )
            self._cache[normalized] = result
            return result

        # Try fuzzy matching
        best_match = None
        best_score = 0.0

        for alias, canonical in self._alias_to_canonical.items():
            score = self._calculate_similarity(normalized, alias)
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = canonical

        if best_match:
            result = EntityMatch(
                matched=True,
                confidence=best_score,
                canonical_name=best_match,
                matched_name=name,
                match_type="FUZZY",
                metadata={
                    "ticker": self.COMPANY_DATA[best_match].get("ticker"),
                    "type": self.COMPANY_DATA[best_match].get("type"),
                },
            )
            self._cache[normalized] = result
            return result

        # No match
        result = EntityMatch(
            matched=False,
            confidence=0.0,
            canonical_name=name,
            matched_name=name,
            match_type="NONE",
        )
        self._cache[normalized] = result
        return result

    def get_ticker(self, name: str) -> Optional[str]:
        """Get ticker symbol for a company name."""
        result = self.resolve(name)
        return result.metadata.get("ticker") if result.matched else None

    def get_company_type(self, name: str) -> Optional[str]:
        """Get company type (BRANDED, GENERIC, BOTH)."""
        result = self.resolve(name)
        return result.metadata.get("type") if result.matched else None


class EntityResolutionService:
    """
    Unified service for entity resolution across all entity types.

    Provides a single interface for resolving drugs, companies, and
    matching entities across data sources.
    """

    def __init__(
        self,
        drug_fuzzy_threshold: float = 0.85,
        company_fuzzy_threshold: float = 0.80,
    ):
        """
        Initialize the entity resolution service.

        Args:
            drug_fuzzy_threshold: Fuzzy threshold for drug matching.
            company_fuzzy_threshold: Fuzzy threshold for company matching.
        """
        self.drug_resolver = DrugNameResolver(fuzzy_threshold=drug_fuzzy_threshold)
        self.company_resolver = CompanyNameResolver(fuzzy_threshold=company_fuzzy_threshold)

    def resolve_drug(self, name: str) -> EntityMatch:
        """Resolve a drug name."""
        return self.drug_resolver.resolve(name)

    def resolve_company(self, name: str) -> EntityMatch:
        """Resolve a company name."""
        return self.company_resolver.resolve(name)

    def match_drugs(self, name1: str, name2: str) -> EntityMatch:
        """Check if two drug names refer to the same drug."""
        return self.drug_resolver.match(name1, name2)

    def get_drug_names(self, name: str) -> List[str]:
        """Get all known names for a drug."""
        result = self.drug_resolver.resolve(name)
        if result.matched:
            return self.drug_resolver.get_all_names(result.canonical_name)
        return [name]

    def get_company_ticker(self, name: str) -> Optional[str]:
        """Get ticker symbol for a company."""
        return self.company_resolver.get_ticker(name)

    def enrich_drug_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a drug record with resolved names.

        Args:
            record: Drug record dictionary.

        Returns:
            Enriched record with canonical names.
        """
        enriched = record.copy()

        # Resolve drug name
        for field in ["brand_name", "generic_name", "active_ingredient"]:
            if record.get(field):
                result = self.resolve_drug(record[field])
                if result.matched:
                    enriched[f"{field}_canonical"] = result.canonical_name
                    enriched[f"{field}_confidence"] = result.confidence

        # Resolve company name
        if record.get("branded_company"):
            company_result = self.resolve_company(record["branded_company"])
            if company_result.matched:
                enriched["branded_company_canonical"] = company_result.canonical_name
                enriched["branded_company_ticker"] = company_result.metadata.get("ticker")
                enriched["company_resolution_confidence"] = company_result.confidence

        return enriched

    def deduplicate_records(
        self,
        records: List[Dict[str, Any]],
        key_field: str = "brand_name",
        entity_type: str = "drug",
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate records based on entity resolution.

        Args:
            records: List of records to deduplicate.
            key_field: Field to use for matching.
            entity_type: Type of entity (drug or company).

        Returns:
            Deduplicated list of records.
        """
        resolver = self.drug_resolver if entity_type == "drug" else self.company_resolver

        # Group by canonical name
        groups: Dict[str, List[Dict[str, Any]]] = {}

        for record in records:
            key_value = record.get(key_field, "")
            result = resolver.resolve(key_value)
            canonical = result.canonical_name if result.matched else key_value

            if canonical not in groups:
                groups[canonical] = []
            groups[canonical].append(record)

        # Take the most complete record from each group
        deduplicated = []
        for canonical, group in groups.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Score each record by completeness
                def completeness_score(rec: Dict) -> int:
                    return sum(1 for v in rec.values() if v is not None and v != "")

                best_record = max(group, key=completeness_score)
                deduplicated.append(best_record)

        logger.info(
            f"Deduplicated {len(records)} records to {len(deduplicated)} unique entities"
        )

        return deduplicated


# Convenience function for creating service
def get_entity_resolution_service() -> EntityResolutionService:
    """Get a pre-configured entity resolution service."""
    return EntityResolutionService()


if __name__ == "__main__":
    # Test entity resolution
    service = get_entity_resolution_service()

    print("=== Drug Name Resolution ===")
    test_drugs = ["Humira", "humira", "HUMIRA", "adalimumab", "Hadlima"]
    for drug in test_drugs:
        result = service.resolve_drug(drug)
        print(f"  {drug} -> {result.canonical_name} ({result.match_type}, {result.confidence:.2f})")

    print("\n=== Drug Matching ===")
    match_result = service.match_drugs("Humira", "adalimumab")
    print(f"  Humira vs adalimumab: matched={match_result.matched}, confidence={match_result.confidence:.2f}")

    print("\n=== Company Resolution ===")
    test_companies = ["AbbVie", "abbvie", "ABBV", "Teva", "Mylan"]
    for company in test_companies:
        result = service.resolve_company(company)
        print(f"  {company} -> {result.canonical_name} (ticker: {result.metadata.get('ticker')})")

    print("\n=== Record Enrichment ===")
    sample_record = {
        "brand_name": "Humira",
        "generic_name": "adalimumab",
        "branded_company": "AbbVie",
    }
    enriched = service.enrich_drug_record(sample_record)
    print(f"  Original: {sample_record}")
    print(f"  Enriched: {enriched}")
