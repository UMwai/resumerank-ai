"""
Patent Cliff Scoring Model

Calculates certainty scores for patent cliff events and generates
trade recommendations based on multiple factors.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScoringWeights:
    """Weights for the certainty score calculation."""

    patent_expired: float = 0.40
    no_litigation: float = 0.30
    anda_approved: float = 0.20
    no_extension: float = 0.10


@dataclass
class DrugPatentData:
    """Data structure for patent cliff analysis."""

    drug_id: int
    brand_name: str
    generic_name: str
    branded_company: str
    branded_company_ticker: Optional[str]
    annual_revenue: Optional[int]

    # Patent information
    patent_numbers: List[str]
    earliest_expiration: Optional[date]
    latest_expiration: Optional[date]
    all_patents_expired: bool
    expiring_patents_count: int
    total_patents_count: int

    # Litigation status
    active_litigation_count: int
    resolved_litigation_count: int
    patents_invalidated: int

    # Generic competition
    approved_generics_count: int
    pending_generics_count: int
    first_to_file_exists: bool

    # Extension potential
    pte_applied: bool
    pediatric_exclusivity: bool


class CertaintyScoreCalculator:
    """
    Calculates patent cliff certainty scores.

    The certainty score (0-100%) indicates how likely a patent cliff event
    will occur as expected, enabling investment decisions.
    """

    def __init__(self, weights: Optional[ScoringWeights] = None):
        """
        Initialize the calculator.

        Args:
            weights: Custom scoring weights. Uses defaults if not provided.
        """
        self.weights = weights or ScoringWeights()

    def calculate_patent_expiration_score(
        self,
        all_expired: bool,
        expiring_count: int,
        total_count: int,
        days_until_expiration: Optional[int] = None,
    ) -> float:
        """
        Calculate the patent expiration component score.

        Args:
            all_expired: Whether all patents have expired.
            expiring_count: Number of patents expiring within 18 months.
            total_count: Total number of patents.
            days_until_expiration: Days until earliest expiration.

        Returns:
            Score from 0 to 100.
        """
        if all_expired:
            return 100.0

        if total_count == 0:
            return 100.0  # No patents means no protection

        # Calculate ratio of expired/expiring patents
        ratio = expiring_count / total_count

        # Apply time decay - closer to expiration = higher score
        if days_until_expiration is not None:
            if days_until_expiration <= 0:
                time_factor = 1.0
            elif days_until_expiration <= 90:  # 3 months
                time_factor = 0.95
            elif days_until_expiration <= 180:  # 6 months
                time_factor = 0.85
            elif days_until_expiration <= 365:  # 1 year
                time_factor = 0.70
            elif days_until_expiration <= 540:  # 18 months
                time_factor = 0.50
            else:
                time_factor = 0.30
        else:
            time_factor = 0.50

        return min(100.0, ratio * 100 * time_factor)

    def calculate_litigation_score(
        self,
        active_litigation: int,
        resolved_litigation: int,
        patents_invalidated: int,
    ) -> float:
        """
        Calculate the litigation status component score.

        Higher score = lower risk (no litigation or favorable outcomes).

        Args:
            active_litigation: Number of ongoing litigation cases.
            resolved_litigation: Number of resolved cases.
            patents_invalidated: Number of patents invalidated by courts.

        Returns:
            Score from 0 to 100.
        """
        # No litigation is best
        if active_litigation == 0:
            # Bonus if patents were previously invalidated
            if patents_invalidated > 0:
                return 100.0
            return 100.0

        # Active litigation reduces certainty significantly
        base_score = 100.0 - (active_litigation * 30)

        # Some credit for resolved litigation (if favorable)
        if resolved_litigation > 0 and patents_invalidated > 0:
            invalidation_rate = patents_invalidated / resolved_litigation
            base_score += invalidation_rate * 20

        return max(0.0, min(100.0, base_score))

    def calculate_anda_score(
        self,
        approved_count: int,
        pending_count: int,
        first_to_file: bool,
    ) -> float:
        """
        Calculate the ANDA approval component score.

        More approved generics = higher certainty of competition.

        Args:
            approved_count: Number of approved ANDAs.
            pending_count: Number of pending ANDAs.
            first_to_file: Whether first-to-file exclusivity exists.

        Returns:
            Score from 0 to 100.
        """
        if approved_count >= 3:
            return 100.0
        elif approved_count >= 2:
            return 85.0
        elif approved_count >= 1:
            return 70.0
        elif pending_count >= 3:
            return 50.0
        elif pending_count >= 1 or first_to_file:
            return 30.0
        else:
            return 0.0

    def calculate_extension_score(
        self,
        pte_applied: bool,
        pediatric_exclusivity: bool,
        drug_class: Optional[str] = None,
    ) -> float:
        """
        Calculate the extension risk component score.

        Lower score if extensions are likely.

        Args:
            pte_applied: Whether PTE has been applied for.
            pediatric_exclusivity: Whether pediatric exclusivity applies.
            drug_class: Drug therapeutic class (affects PTE likelihood).

        Returns:
            Score from 0 to 100.
        """
        score = 100.0

        if pte_applied:
            score -= 30  # PTE application reduces certainty

        if pediatric_exclusivity:
            score -= 20  # Pediatric exclusivity adds 6 months

        # Certain drug classes more likely to get extensions
        if drug_class:
            high_extension_classes = ["oncology", "rare disease", "orphan"]
            if any(cls in drug_class.lower() for cls in high_extension_classes):
                score -= 15

        return max(0.0, score)

    def calculate_certainty_score(self, data: DrugPatentData) -> Dict[str, Any]:
        """
        Calculate the overall certainty score for a patent cliff event.

        Args:
            data: DrugPatentData with all relevant information.

        Returns:
            Dictionary with score breakdown and final score.
        """
        # Calculate component scores
        days_until = None
        if data.earliest_expiration:
            days_until = (data.earliest_expiration - date.today()).days

        patent_score = self.calculate_patent_expiration_score(
            all_expired=data.all_patents_expired,
            expiring_count=data.expiring_patents_count,
            total_count=data.total_patents_count,
            days_until_expiration=days_until,
        )

        litigation_score = self.calculate_litigation_score(
            active_litigation=data.active_litigation_count,
            resolved_litigation=data.resolved_litigation_count,
            patents_invalidated=data.patents_invalidated,
        )

        anda_score = self.calculate_anda_score(
            approved_count=data.approved_generics_count,
            pending_count=data.pending_generics_count,
            first_to_file=data.first_to_file_exists,
        )

        extension_score = self.calculate_extension_score(
            pte_applied=data.pte_applied,
            pediatric_exclusivity=data.pediatric_exclusivity,
        )

        # Weighted average
        final_score = (
            self.weights.patent_expired * patent_score
            + self.weights.no_litigation * litigation_score
            + self.weights.anda_approved * anda_score
            + self.weights.no_extension * extension_score
        )

        return {
            "drug_id": data.drug_id,
            "brand_name": data.brand_name,
            "patent_expiration_score": round(patent_score, 2),
            "litigation_score": round(litigation_score, 2),
            "anda_score": round(anda_score, 2),
            "extension_score": round(extension_score, 2),
            "final_certainty_score": round(final_score, 2),
            "weights_used": {
                "patent_expired": self.weights.patent_expired,
                "no_litigation": self.weights.no_litigation,
                "anda_approved": self.weights.anda_approved,
                "no_extension": self.weights.no_extension,
            },
        }


class PatentCliffScorer:
    """
    Main scoring class for patent cliff events.

    Combines certainty scoring with market opportunity analysis
    to generate trade recommendations.
    """

    # Thresholds for recommendations
    HIGH_CERTAINTY_THRESHOLD = 80
    MEDIUM_CERTAINTY_THRESHOLD = 60

    # Market opportunity thresholds
    BLOCKBUSTER_THRESHOLD = 1_000_000_000  # $1B
    HIGH_VALUE_THRESHOLD = 500_000_000  # $500M
    MEDIUM_VALUE_THRESHOLD = 100_000_000  # $100M

    # Generic erosion rates
    EROSION_RATES = {
        0: 0.0,
        1: 0.50,
        2: 0.65,
        3: 0.75,
        "4+": 0.80,
    }

    def __init__(self, weights: Optional[ScoringWeights] = None):
        """
        Initialize the scorer.

        Args:
            weights: Custom scoring weights.
        """
        self.certainty_calculator = CertaintyScoreCalculator(weights)

    def calculate_market_opportunity(
        self, annual_revenue: int, num_generics: int
    ) -> Dict[str, Any]:
        """
        Calculate market opportunity for generic entry.

        Args:
            annual_revenue: Current branded drug annual revenue.
            num_generics: Number of expected generic competitors.

        Returns:
            Dictionary with market opportunity analysis.
        """
        if num_generics == 0:
            erosion_rate = 0.50  # Assume at least 50% erosion with first generic
        elif num_generics >= 4:
            erosion_rate = self.EROSION_RATES["4+"]
        else:
            erosion_rate = self.EROSION_RATES[num_generics]

        revenue_at_risk = int(annual_revenue * erosion_rate)
        generic_market_size = revenue_at_risk

        # Classify opportunity tier
        if generic_market_size >= self.BLOCKBUSTER_THRESHOLD:
            tier = "BLOCKBUSTER"
        elif generic_market_size >= self.HIGH_VALUE_THRESHOLD:
            tier = "HIGH_VALUE"
        elif generic_market_size >= self.MEDIUM_VALUE_THRESHOLD:
            tier = "MEDIUM_VALUE"
        else:
            tier = "SMALL"

        return {
            "annual_revenue": annual_revenue,
            "erosion_rate": erosion_rate,
            "revenue_at_risk": revenue_at_risk,
            "generic_market_size": generic_market_size,
            "opportunity_tier": tier,
            "per_generic_opportunity": (
                generic_market_size // max(1, num_generics)
            ),
        }

    def generate_trade_recommendation(
        self,
        certainty_score: float,
        market_opportunity: Dict[str, Any],
        days_until_event: int,
        branded_ticker: Optional[str] = None,
        generic_tickers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate trade recommendation based on analysis.

        Args:
            certainty_score: Patent cliff certainty score (0-100).
            market_opportunity: Market opportunity analysis.
            days_until_event: Days until patent expiration.
            branded_ticker: Branded company stock ticker.
            generic_tickers: List of generic company tickers.

        Returns:
            Trade recommendation dictionary.
        """
        tier = market_opportunity["opportunity_tier"]

        # Skip small opportunities
        if tier == "SMALL":
            return {
                "recommendation": "SKIP",
                "confidence": "N/A",
                "rationale": "Market opportunity too small (<$100M)",
                "trade_type": None,
            }

        # Determine confidence level
        if certainty_score >= self.HIGH_CERTAINTY_THRESHOLD:
            confidence = "HIGH"
        elif certainty_score >= self.MEDIUM_CERTAINTY_THRESHOLD:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Generate recommendation based on certainty and timing
        if confidence == "HIGH":
            if days_until_event <= 180:  # 6 months
                recommendation = "EXECUTE TRADE"
                trade_type = "PAIR_TRADE"
                rationale = (
                    f"High certainty ({certainty_score:.0f}%) patent cliff in "
                    f"{days_until_event} days. {tier} opportunity "
                    f"(${market_opportunity['revenue_at_risk']:,} at risk)."
                )
            elif days_until_event <= 365:  # 12 months
                recommendation = "INITIATE POSITION"
                trade_type = "PAIR_TRADE"
                rationale = (
                    f"High certainty ({certainty_score:.0f}%) patent cliff in "
                    f"{days_until_event//30} months. Begin building position."
                )
            else:
                recommendation = "MONITOR"
                trade_type = None
                rationale = (
                    f"High certainty but event >12 months out. Add to watchlist."
                )

        elif confidence == "MEDIUM":
            recommendation = "MONITOR CLOSELY"
            trade_type = None
            rationale = (
                f"Medium certainty ({certainty_score:.0f}%). "
                f"Wait for resolution of uncertainties before trading."
            )

        else:
            recommendation = "DO NOT TRADE"
            trade_type = None
            rationale = (
                f"Low certainty ({certainty_score:.0f}%). "
                f"Too many unknowns for profitable trade."
            )

        result = {
            "recommendation": recommendation,
            "confidence": confidence,
            "rationale": rationale,
            "trade_type": trade_type,
            "certainty_score": certainty_score,
            "market_opportunity": market_opportunity["revenue_at_risk"],
            "opportunity_tier": tier,
            "days_until_event": days_until_event,
        }

        # Add specific trade suggestions
        if trade_type == "PAIR_TRADE":
            result["suggested_trades"] = {
                "short": {
                    "ticker": branded_ticker,
                    "action": "SHORT",
                    "rationale": "Branded company faces revenue erosion",
                },
                "long": {
                    "tickers": generic_tickers or [],
                    "action": "LONG",
                    "rationale": "Generic companies to capture market share",
                },
            }

        return result

    def score_patent_cliff(self, data: DrugPatentData) -> Dict[str, Any]:
        """
        Comprehensive scoring of a patent cliff event.

        Args:
            data: DrugPatentData with all relevant information.

        Returns:
            Complete analysis with scores and recommendations.
        """
        # Calculate certainty score
        certainty_result = self.certainty_calculator.calculate_certainty_score(data)
        certainty_score = certainty_result["final_certainty_score"]

        # Calculate market opportunity
        annual_revenue = data.annual_revenue or 0
        num_generics = max(1, data.approved_generics_count + data.pending_generics_count)
        market_opportunity = self.calculate_market_opportunity(annual_revenue, num_generics)

        # Calculate days until event
        if data.earliest_expiration:
            days_until = (data.earliest_expiration - date.today()).days
        else:
            days_until = 999999

        # Generate trade recommendation
        trade_rec = self.generate_trade_recommendation(
            certainty_score=certainty_score,
            market_opportunity=market_opportunity,
            days_until_event=days_until,
            branded_ticker=data.branded_company_ticker,
        )

        return {
            "drug": {
                "drug_id": data.drug_id,
                "brand_name": data.brand_name,
                "generic_name": data.generic_name,
                "branded_company": data.branded_company,
                "branded_company_ticker": data.branded_company_ticker,
            },
            "patent_info": {
                "earliest_expiration": data.earliest_expiration,
                "latest_expiration": data.latest_expiration,
                "total_patents": data.total_patents_count,
                "days_until_earliest_expiration": days_until,
            },
            "competition": {
                "approved_generics": data.approved_generics_count,
                "pending_generics": data.pending_generics_count,
                "active_litigation": data.active_litigation_count,
            },
            "scoring": certainty_result,
            "market_opportunity": market_opportunity,
            "trade_recommendation": trade_rec,
        }


if __name__ == "__main__":
    # Test the scoring model
    print("\n=== Testing Patent Cliff Scoring Model ===")

    # Create test data
    test_drug = DrugPatentData(
        drug_id=1,
        brand_name="Humira",
        generic_name="adalimumab",
        branded_company="AbbVie Inc.",
        branded_company_ticker="ABBV",
        annual_revenue=20_000_000_000,  # $20B
        patent_numbers=["US6090382", "US6258562"],
        earliest_expiration=date(2024, 12, 31),
        latest_expiration=date(2025, 6, 30),
        all_patents_expired=False,
        expiring_patents_count=2,
        total_patents_count=2,
        active_litigation_count=0,
        resolved_litigation_count=3,
        patents_invalidated=2,
        approved_generics_count=8,
        pending_generics_count=2,
        first_to_file_exists=True,
        pte_applied=False,
        pediatric_exclusivity=False,
    )

    scorer = PatentCliffScorer()
    result = scorer.score_patent_cliff(test_drug)

    print(f"\nDrug: {result['drug']['brand_name']} ({result['drug']['generic_name']})")
    print(f"Company: {result['drug']['branded_company']} ({result['drug']['branded_company_ticker']})")

    print(f"\nCertainty Score: {result['scoring']['final_certainty_score']:.1f}%")
    print(f"  - Patent Score: {result['scoring']['patent_expiration_score']:.1f}")
    print(f"  - Litigation Score: {result['scoring']['litigation_score']:.1f}")
    print(f"  - ANDA Score: {result['scoring']['anda_score']:.1f}")
    print(f"  - Extension Score: {result['scoring']['extension_score']:.1f}")

    print(f"\nMarket Opportunity: ${result['market_opportunity']['revenue_at_risk']:,}")
    print(f"Opportunity Tier: {result['market_opportunity']['opportunity_tier']}")

    print(f"\nTrade Recommendation: {result['trade_recommendation']['recommendation']}")
    print(f"Confidence: {result['trade_recommendation']['confidence']}")
    print(f"Rationale: {result['trade_recommendation']['rationale']}")
