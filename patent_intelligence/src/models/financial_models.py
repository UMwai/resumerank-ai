"""
Financial Modeling for Patent Cliff Events

Models revenue erosion curves, generic market share, and provides
risk-adjusted NPV calculations for trade sizing decisions.

Features:
- Revenue erosion modeling (Years 1-5 post-generic)
- Generic market share projections
- Risk-adjusted NPV calculations
- Options pricing strategy suggestions
- Trade size recommendations
"""

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ErosionForecast:
    """Revenue erosion forecast for a drug."""

    drug_name: str
    brand_company: str
    base_revenue: int  # Pre-generic annual revenue

    # Yearly projections
    year_1_revenue: int
    year_2_revenue: int
    year_3_revenue: int
    year_4_revenue: int
    year_5_revenue: int

    # Erosion rates
    year_1_erosion_pct: float
    year_2_erosion_pct: float
    year_3_erosion_pct: float
    year_4_erosion_pct: float
    year_5_erosion_pct: float

    # Total impact
    total_revenue_loss_5yr: int
    cumulative_erosion_pct: float

    # Generic market
    generic_market_year_1: int
    generic_market_year_5: int
    num_generic_competitors_projected: int

    # Confidence
    forecast_confidence: str  # low, medium, high
    scenario: str  # conservative, base, aggressive

    forecast_date: date = field(default_factory=date.today)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["forecast_date"] = self.forecast_date.isoformat()
        return result


@dataclass
class NPVAnalysis:
    """Net Present Value analysis for trade decision."""

    drug_name: str

    # NPV calculations
    npv_branded_decline: int  # NPV of branded revenue decline
    npv_generic_opportunity: int  # NPV of generic market
    risk_adjusted_npv: int  # NPV adjusted for litigation risk

    # Trade sizing
    recommended_position_size_pct: float  # % of portfolio
    recommended_notional: int  # Dollar amount
    max_loss_estimate: int

    # Risk metrics
    win_probability: float
    expected_value: int
    sharpe_ratio: float

    # Scenarios
    bull_case_return: float
    base_case_return: float
    bear_case_return: float

    # Options analysis
    suggested_options_strategy: str
    implied_volatility_estimate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class RevenueErosionModel:
    """
    Models revenue erosion following patent cliff/generic entry.

    Uses historical data and market dynamics to project revenue
    decline curves for branded drugs facing generic competition.
    """

    # Historical erosion curves by therapeutic area
    # Format: (year1, year2, year3, year4, year5) - cumulative erosion
    EROSION_CURVES = {
        "small_molecule_single": {
            "conservative": (0.30, 0.55, 0.70, 0.80, 0.85),
            "base": (0.40, 0.65, 0.78, 0.85, 0.90),
            "aggressive": (0.55, 0.75, 0.85, 0.90, 0.93),
        },
        "small_molecule_multiple": {  # Multiple generic entrants
            "conservative": (0.45, 0.70, 0.82, 0.88, 0.92),
            "base": (0.55, 0.78, 0.87, 0.92, 0.95),
            "aggressive": (0.65, 0.85, 0.92, 0.95, 0.97),
        },
        "biologic": {  # Biosimilars - slower erosion
            "conservative": (0.15, 0.30, 0.45, 0.55, 0.62),
            "base": (0.25, 0.42, 0.55, 0.65, 0.72),
            "aggressive": (0.35, 0.52, 0.65, 0.75, 0.80),
        },
        "complex_generic": {  # Complex drugs (inhalers, etc.)
            "conservative": (0.20, 0.38, 0.52, 0.62, 0.70),
            "base": (0.30, 0.48, 0.62, 0.72, 0.78),
            "aggressive": (0.40, 0.58, 0.72, 0.80, 0.85),
        },
    }

    # Therapeutic area adjustments
    THERAPEUTIC_ADJUSTMENTS = {
        "Oncology": -0.05,  # Slower erosion - specialty
        "Immunology": -0.03,
        "Cardiovascular": 0.05,  # Faster erosion - commoditized
        "Diabetes": 0.03,
        "Neurology": 0.02,
        "Ophthalmology": -0.02,
    }

    # Generic competitor impact
    COMPETITOR_MULTIPLIERS = {
        1: 0.85,  # Single generic - less erosion
        2: 0.95,
        3: 1.00,  # Base case
        4: 1.05,
        "5+": 1.10,  # Many generics - faster erosion
    }

    def __init__(self):
        """Initialize the erosion model."""
        logger.info("Revenue erosion model initialized")

    def determine_drug_type(
        self,
        is_biologic: bool,
        is_complex: bool,
        num_competitors: int,
    ) -> str:
        """Determine the drug type category for erosion curve selection."""
        if is_biologic:
            return "biologic"
        elif is_complex:
            return "complex_generic"
        elif num_competitors >= 3:
            return "small_molecule_multiple"
        else:
            return "small_molecule_single"

    def forecast_erosion(
        self,
        drug_name: str,
        brand_company: str,
        base_revenue: int,
        therapeutic_area: str = "Other",
        is_biologic: bool = False,
        is_complex: bool = False,
        num_generic_competitors: int = 3,
        patent_strength_score: float = 50.0,
        scenario: str = "base",
    ) -> ErosionForecast:
        """
        Forecast revenue erosion over 5 years post-generic entry.

        Args:
            drug_name: Name of the branded drug.
            brand_company: Branded drug company.
            base_revenue: Pre-generic annual revenue.
            therapeutic_area: Drug therapeutic area.
            is_biologic: Whether drug is a biologic.
            is_complex: Whether drug is a complex generic.
            num_generic_competitors: Expected number of generic competitors.
            patent_strength_score: Patent strength (0-100).
            scenario: conservative, base, or aggressive.

        Returns:
            ErosionForecast with yearly projections.
        """
        # Select erosion curve
        drug_type = self.determine_drug_type(is_biologic, is_complex, num_generic_competitors)
        curves = self.EROSION_CURVES[drug_type]
        erosion_curve = curves[scenario]

        # Apply therapeutic area adjustment
        adjustment = self.THERAPEUTIC_ADJUSTMENTS.get(therapeutic_area, 0.0)

        # Apply competitor multiplier
        if num_generic_competitors >= 5:
            multiplier = self.COMPETITOR_MULTIPLIERS["5+"]
        else:
            multiplier = self.COMPETITOR_MULTIPLIERS.get(num_generic_competitors, 1.0)

        # Adjust erosion rates
        adjusted_erosion = []
        for rate in erosion_curve:
            adjusted_rate = min(0.98, max(0.1, (rate + adjustment) * multiplier))
            adjusted_erosion.append(adjusted_rate)

        # Calculate yearly revenues
        year_revenues = []
        for erosion in adjusted_erosion:
            year_revenue = int(base_revenue * (1 - erosion))
            year_revenues.append(year_revenue)

        # Calculate generic market size
        generic_market_year_1 = int(base_revenue * adjusted_erosion[0] * 0.3)  # 30% of lost revenue
        generic_market_year_5 = int(base_revenue * adjusted_erosion[4] * 0.25)  # Price erosion

        # Total revenue loss
        total_loss = sum(base_revenue - yr for yr in year_revenues)

        # Determine confidence based on inputs
        if patent_strength_score > 70 or scenario == "aggressive":
            confidence = "high"
        elif patent_strength_score < 40 or scenario == "conservative":
            confidence = "low"
        else:
            confidence = "medium"

        return ErosionForecast(
            drug_name=drug_name,
            brand_company=brand_company,
            base_revenue=base_revenue,
            year_1_revenue=year_revenues[0],
            year_2_revenue=year_revenues[1],
            year_3_revenue=year_revenues[2],
            year_4_revenue=year_revenues[3],
            year_5_revenue=year_revenues[4],
            year_1_erosion_pct=round(adjusted_erosion[0] * 100, 1),
            year_2_erosion_pct=round(adjusted_erosion[1] * 100, 1),
            year_3_erosion_pct=round(adjusted_erosion[2] * 100, 1),
            year_4_erosion_pct=round(adjusted_erosion[3] * 100, 1),
            year_5_erosion_pct=round(adjusted_erosion[4] * 100, 1),
            total_revenue_loss_5yr=total_loss,
            cumulative_erosion_pct=round(adjusted_erosion[4] * 100, 1),
            generic_market_year_1=generic_market_year_1,
            generic_market_year_5=generic_market_year_5,
            num_generic_competitors_projected=num_generic_competitors,
            forecast_confidence=confidence,
            scenario=scenario,
        )

    def multi_scenario_forecast(
        self,
        drug_name: str,
        brand_company: str,
        base_revenue: int,
        **kwargs,
    ) -> Dict[str, ErosionForecast]:
        """
        Generate forecasts for all three scenarios.

        Returns:
            Dict mapping scenario name to ErosionForecast.
        """
        forecasts = {}

        for scenario in ["conservative", "base", "aggressive"]:
            forecasts[scenario] = self.forecast_erosion(
                drug_name=drug_name,
                brand_company=brand_company,
                base_revenue=base_revenue,
                scenario=scenario,
                **kwargs,
            )

        return forecasts


class GenericMarketModel:
    """
    Models generic market dynamics and market share evolution.
    """

    # First-to-file exclusivity parameters
    FTF_EXCLUSIVITY_MONTHS = 6
    FTF_MARKET_SHARE_INITIAL = 0.70  # First generic gets 70% of generic market

    # Price erosion by competitor count
    PRICE_EROSION_BY_COMPETITORS = {
        1: 0.80,  # Price at 80% of branded
        2: 0.55,  # Price at 55% of branded
        3: 0.40,
        4: 0.30,
        "5+": 0.20,  # Commodity pricing
    }

    def project_generic_market_share(
        self,
        base_revenue: int,
        months_since_generic_entry: int,
        num_competitors: int,
        is_biologic: bool = False,
    ) -> Dict[str, Any]:
        """
        Project generic market share at a given time point.

        Args:
            base_revenue: Original branded revenue.
            months_since_generic_entry: Months since first generic.
            num_competitors: Number of generic competitors.
            is_biologic: Whether biosimilar.

        Returns:
            Dict with market share projections.
        """
        # Calculate market capture rate
        if is_biologic:
            # Biosimilars have slower uptake
            if months_since_generic_entry <= 12:
                market_capture = 0.20 + (months_since_generic_entry / 12) * 0.15
            elif months_since_generic_entry <= 36:
                market_capture = 0.35 + ((months_since_generic_entry - 12) / 24) * 0.25
            else:
                market_capture = min(0.70, 0.60 + ((months_since_generic_entry - 36) / 24) * 0.10)
        else:
            # Small molecules have rapid uptake
            if months_since_generic_entry <= 6:
                market_capture = 0.40 + (months_since_generic_entry / 6) * 0.20
            elif months_since_generic_entry <= 24:
                market_capture = 0.60 + ((months_since_generic_entry - 6) / 18) * 0.25
            else:
                market_capture = min(0.95, 0.85 + ((months_since_generic_entry - 24) / 36) * 0.10)

        # Price level
        if num_competitors >= 5:
            price_level = self.PRICE_EROSION_BY_COMPETITORS["5+"]
        else:
            price_level = self.PRICE_EROSION_BY_COMPETITORS.get(num_competitors, 0.40)

        # Generic market value
        generic_volume_value = base_revenue * market_capture
        generic_revenue_value = generic_volume_value * price_level

        # Branded remaining
        branded_remaining = base_revenue * (1 - market_capture)

        return {
            "months_since_entry": months_since_generic_entry,
            "generic_market_capture_pct": round(market_capture * 100, 1),
            "branded_market_share_pct": round((1 - market_capture) * 100, 1),
            "generic_price_vs_branded_pct": round(price_level * 100, 1),
            "generic_revenue_annual": int(generic_revenue_value),
            "branded_revenue_annual": int(branded_remaining),
            "total_market_value": int(generic_revenue_value + branded_remaining),
            "market_value_decline_pct": round((1 - (generic_revenue_value + branded_remaining) / base_revenue) * 100, 1),
        }


class NPVCalculator:
    """
    Calculates Net Present Value for patent cliff trading decisions.
    """

    DEFAULT_DISCOUNT_RATE = 0.10  # 10% annual
    DEFAULT_RISK_FREE_RATE = 0.05  # 5% risk-free rate

    def __init__(self, discount_rate: float = DEFAULT_DISCOUNT_RATE):
        """
        Initialize NPV calculator.

        Args:
            discount_rate: Annual discount rate for NPV calculations.
        """
        self.discount_rate = discount_rate

    def calculate_npv(
        self,
        cash_flows: List[int],
        years: Optional[List[int]] = None,
    ) -> int:
        """
        Calculate Net Present Value of cash flows.

        Args:
            cash_flows: List of annual cash flows.
            years: Optional list of year numbers (default: 1, 2, 3...).

        Returns:
            NPV as integer.
        """
        if years is None:
            years = list(range(1, len(cash_flows) + 1))

        npv = 0
        for cf, year in zip(cash_flows, years):
            npv += cf / ((1 + self.discount_rate) ** year)

        return int(npv)

    def calculate_risk_adjusted_npv(
        self,
        drug_name: str,
        erosion_forecast: ErosionForecast,
        upheld_probability: float,
        portfolio_value: int = 1000000,
        risk_tolerance: str = "moderate",
    ) -> NPVAnalysis:
        """
        Calculate risk-adjusted NPV and trading recommendations.

        Args:
            drug_name: Drug name.
            erosion_forecast: Erosion forecast from RevenueErosionModel.
            upheld_probability: Probability patent is upheld (0-100).
            portfolio_value: Total portfolio value for sizing.
            risk_tolerance: conservative, moderate, aggressive.

        Returns:
            NPVAnalysis with recommendations.
        """
        # Cash flows from branded perspective (losses)
        branded_losses = [
            erosion_forecast.base_revenue - erosion_forecast.year_1_revenue,
            erosion_forecast.base_revenue - erosion_forecast.year_2_revenue,
            erosion_forecast.base_revenue - erosion_forecast.year_3_revenue,
            erosion_forecast.base_revenue - erosion_forecast.year_4_revenue,
            erosion_forecast.base_revenue - erosion_forecast.year_5_revenue,
        ]

        # NPV of branded decline
        npv_branded_decline = self.calculate_npv(branded_losses)

        # NPV of generic opportunity (portion of market)
        generic_share = 0.15  # Assume we can capture 15% of generic market
        generic_revenues = [
            int(erosion_forecast.generic_market_year_1 * generic_share),
            int(erosion_forecast.generic_market_year_1 * generic_share * 0.8),  # Price erosion
            int(erosion_forecast.generic_market_year_5 * generic_share * 0.9),
            int(erosion_forecast.generic_market_year_5 * generic_share * 0.95),
            int(erosion_forecast.generic_market_year_5 * generic_share),
        ]
        npv_generic_opportunity = self.calculate_npv(generic_revenues)

        # Win probability (inverse of upheld probability for short trade)
        win_prob = (100 - upheld_probability) / 100

        # Risk-adjusted NPV
        risk_adjusted_npv = int(npv_branded_decline * win_prob)

        # Expected value
        expected_value = int(npv_branded_decline * win_prob - npv_branded_decline * 0.2 * (1 - win_prob))

        # Position sizing based on risk tolerance
        sizing_multipliers = {
            "conservative": 0.02,
            "moderate": 0.05,
            "aggressive": 0.10,
        }
        sizing_mult = sizing_multipliers.get(risk_tolerance, 0.05)

        # Scale position by confidence
        confidence_scale = abs(win_prob - 0.5) * 2  # 0 at 50%, 1 at 0% or 100%
        position_size_pct = sizing_mult * (0.5 + confidence_scale * 0.5) * 100

        recommended_notional = int(portfolio_value * position_size_pct / 100)
        max_loss_estimate = int(recommended_notional * 0.3)  # Assume 30% max loss

        # Sharpe ratio estimate
        expected_return = win_prob * 0.25 - (1 - win_prob) * 0.15
        volatility = 0.35  # Estimated volatility
        sharpe_ratio = (expected_return - self.DEFAULT_RISK_FREE_RATE) / volatility

        # Scenario returns
        bull_case_return = 0.40  # 40% return if patent falls
        base_case_return = expected_return * 100
        bear_case_return = -0.20  # 20% loss if patent upheld

        # Options strategy
        if upheld_probability < 40:
            options_strategy = "Long put spreads on branded company"
            iv_estimate = 0.40
        elif upheld_probability < 60:
            options_strategy = "Iron condor to capture volatility"
            iv_estimate = 0.35
        else:
            options_strategy = "Collar on long generic position"
            iv_estimate = 0.30

        return NPVAnalysis(
            drug_name=drug_name,
            npv_branded_decline=npv_branded_decline,
            npv_generic_opportunity=npv_generic_opportunity,
            risk_adjusted_npv=risk_adjusted_npv,
            recommended_position_size_pct=round(position_size_pct, 2),
            recommended_notional=recommended_notional,
            max_loss_estimate=max_loss_estimate,
            win_probability=round(win_prob * 100, 1),
            expected_value=expected_value,
            sharpe_ratio=round(sharpe_ratio, 2),
            bull_case_return=round(bull_case_return * 100, 1),
            base_case_return=round(base_case_return, 1),
            bear_case_return=round(bear_case_return * 100, 1),
            suggested_options_strategy=options_strategy,
            implied_volatility_estimate=round(iv_estimate * 100, 1),
        )


class TradeRecommendationEngine:
    """
    Generates trade recommendations based on financial models.
    """

    def __init__(self):
        """Initialize the recommendation engine."""
        self.erosion_model = RevenueErosionModel()
        self.market_model = GenericMarketModel()
        self.npv_calculator = NPVCalculator()

    def generate_recommendation(
        self,
        drug_name: str,
        brand_company: str,
        brand_ticker: str,
        base_revenue: int,
        therapeutic_area: str,
        is_biologic: bool,
        num_competitors: int,
        upheld_probability: float,
        days_until_event: int,
        portfolio_value: int = 1000000,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive trade recommendation.

        Args:
            drug_name: Drug name.
            brand_company: Branded company.
            brand_ticker: Stock ticker.
            base_revenue: Annual revenue.
            therapeutic_area: Therapeutic area.
            is_biologic: Whether biologic.
            num_competitors: Expected generic competitors.
            upheld_probability: Probability patent upheld.
            days_until_event: Days until patent cliff.
            portfolio_value: Portfolio value for sizing.

        Returns:
            Comprehensive recommendation dict.
        """
        # Generate erosion forecast
        erosion = self.erosion_model.forecast_erosion(
            drug_name=drug_name,
            brand_company=brand_company,
            base_revenue=base_revenue,
            therapeutic_area=therapeutic_area,
            is_biologic=is_biologic,
            num_generic_competitors=num_competitors,
        )

        # Calculate NPV and position sizing
        npv_analysis = self.npv_calculator.calculate_risk_adjusted_npv(
            drug_name=drug_name,
            erosion_forecast=erosion,
            upheld_probability=upheld_probability,
            portfolio_value=portfolio_value,
        )

        # Determine action
        if upheld_probability < 40 and days_until_event < 180:
            action = "EXECUTE"
            action_rationale = f"High probability ({100-upheld_probability:.0f}%) of patent invalidation within 6 months"
        elif upheld_probability < 50 and days_until_event < 365:
            action = "INITIATE"
            action_rationale = "Favorable odds with sufficient time to event"
        elif upheld_probability < 60:
            action = "MONITOR"
            action_rationale = "Marginal odds - wait for additional catalysts"
        else:
            action = "AVOID"
            action_rationale = f"High probability ({upheld_probability:.0f}%) patent will be upheld"

        # Trade structure
        trade_structure = {
            "primary": {
                "instrument": f"{brand_ticker} stock",
                "direction": "SHORT" if action in ["EXECUTE", "INITIATE"] else "NONE",
                "size": f"${npv_analysis.recommended_notional:,}",
                "entry_timing": "Immediate" if action == "EXECUTE" else "On weakness",
            },
            "hedge": {
                "instrument": f"{brand_ticker} calls",
                "direction": "LONG" if action in ["EXECUTE", "INITIATE"] else "NONE",
                "size": f"{int(npv_analysis.recommended_notional * 0.1):,}",
                "strike": "10% OTM",
            },
            "alternative": {
                "strategy": npv_analysis.suggested_options_strategy,
                "max_loss": f"${npv_analysis.max_loss_estimate:,}",
            },
        }

        return {
            "drug_name": drug_name,
            "brand_company": brand_company,
            "brand_ticker": brand_ticker,
            "action": action,
            "action_rationale": action_rationale,
            "confidence": erosion.forecast_confidence,
            "timing": {
                "days_until_event": days_until_event,
                "event_date": (date.today() + timedelta(days=days_until_event)).isoformat(),
            },
            "financials": {
                "base_revenue": f"${base_revenue:,}",
                "year_1_revenue_forecast": f"${erosion.year_1_revenue:,}",
                "year_5_revenue_forecast": f"${erosion.year_5_revenue:,}",
                "total_revenue_at_risk_5yr": f"${erosion.total_revenue_loss_5yr:,}",
            },
            "npv_analysis": {
                "npv_branded_decline": f"${npv_analysis.npv_branded_decline:,}",
                "risk_adjusted_npv": f"${npv_analysis.risk_adjusted_npv:,}",
                "expected_value": f"${npv_analysis.expected_value:,}",
            },
            "position_sizing": {
                "recommended_size_pct": f"{npv_analysis.recommended_position_size_pct:.2f}%",
                "recommended_notional": f"${npv_analysis.recommended_notional:,}",
                "max_loss": f"${npv_analysis.max_loss_estimate:,}",
            },
            "risk_metrics": {
                "win_probability": f"{npv_analysis.win_probability:.1f}%",
                "sharpe_ratio": npv_analysis.sharpe_ratio,
                "bull_case_return": f"{npv_analysis.bull_case_return:.1f}%",
                "base_case_return": f"{npv_analysis.base_case_return:.1f}%",
                "bear_case_return": f"{npv_analysis.bear_case_return:.1f}%",
            },
            "trade_structure": trade_structure,
            "erosion_forecast": erosion.to_dict(),
        }


def generate_financial_template() -> str:
    """
    Generate a financial modeling template for Excel export.

    Returns:
        CSV-formatted template string.
    """
    template = """Drug Analysis Financial Model Template
Generated: {date}

=== INPUTS ===
Drug Name,
Brand Company,
Brand Ticker,
Base Annual Revenue ($),
Therapeutic Area,
Is Biologic (Y/N),
Expected Generic Competitors,
Patent Upheld Probability (%),
Days Until Patent Cliff,
Portfolio Value ($),

=== EROSION ASSUMPTIONS ===
Year 1 Erosion Rate (%),
Year 2 Erosion Rate (%),
Year 3 Erosion Rate (%),
Year 4 Erosion Rate (%),
Year 5 Erosion Rate (%),

=== REVENUE PROJECTIONS ===
Year,Base Revenue,Eroded Revenue,Revenue Loss,Generic Market
0,{base_revenue},{base_revenue},0,0
1,,,,
2,,,,
3,,,,
4,,,,
5,,,,

=== NPV ANALYSIS ===
Discount Rate (%),10
NPV of Branded Decline,$
NPV of Generic Opportunity,$
Risk-Adjusted NPV,$
Win Probability (%),
Expected Value,$

=== POSITION SIZING ===
Risk Tolerance,Moderate
Position Size (%),
Notional Amount,$
Max Loss Estimate,$

=== SCENARIO ANALYSIS ===
Scenario,Revenue Y1,Revenue Y5,Total Loss,NPV
Conservative,,,,
Base Case,,,,
Aggressive,,,,

=== TRADE RECOMMENDATION ===
Action,
Rationale,
Primary Trade,
Hedge,
Options Strategy,
""".format(date=date.today().isoformat(), base_revenue=10000000000)

    return template


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Financial Modeling - Revenue Erosion and Trade Sizing")
    print("=" * 70)

    # Initialize models
    engine = TradeRecommendationEngine()

    # Test cases
    test_drugs = [
        {
            "drug_name": "Humira",
            "brand_company": "AbbVie Inc.",
            "brand_ticker": "ABBV",
            "base_revenue": 21237000000,
            "therapeutic_area": "Immunology",
            "is_biologic": True,
            "num_competitors": 8,
            "upheld_probability": 35.0,
            "days_until_event": 90,
        },
        {
            "drug_name": "Eliquis",
            "brand_company": "Bristol-Myers Squibb",
            "brand_ticker": "BMY",
            "base_revenue": 12200000000,
            "therapeutic_area": "Cardiovascular",
            "is_biologic": False,
            "num_competitors": 5,
            "upheld_probability": 65.0,
            "days_until_event": 540,
        },
        {
            "drug_name": "Keytruda",
            "brand_company": "Merck & Co.",
            "brand_ticker": "MRK",
            "base_revenue": 25000000000,
            "therapeutic_area": "Oncology",
            "is_biologic": True,
            "num_competitors": 3,
            "upheld_probability": 75.0,
            "days_until_event": 1095,
        },
    ]

    for drug in test_drugs:
        recommendation = engine.generate_recommendation(**drug)

        print(f"\n{'=' * 60}")
        print(f"Drug: {recommendation['drug_name']} ({recommendation['brand_ticker']})")
        print(f"{'=' * 60}")
        print(f"Action: {recommendation['action']}")
        print(f"Rationale: {recommendation['action_rationale']}")
        print(f"\nFinancials:")
        print(f"  Base Revenue: {recommendation['financials']['base_revenue']}")
        print(f"  Year 1 Forecast: {recommendation['financials']['year_1_revenue_forecast']}")
        print(f"  5-Year Revenue at Risk: {recommendation['financials']['total_revenue_at_risk_5yr']}")
        print(f"\nNPV Analysis:")
        print(f"  NPV Branded Decline: {recommendation['npv_analysis']['npv_branded_decline']}")
        print(f"  Risk-Adjusted NPV: {recommendation['npv_analysis']['risk_adjusted_npv']}")
        print(f"\nPosition Sizing:")
        print(f"  Recommended Size: {recommendation['position_sizing']['recommended_size_pct']}")
        print(f"  Notional: {recommendation['position_sizing']['recommended_notional']}")
        print(f"  Max Loss: {recommendation['position_sizing']['max_loss']}")
        print(f"\nRisk Metrics:")
        print(f"  Win Probability: {recommendation['risk_metrics']['win_probability']}")
        print(f"  Sharpe Ratio: {recommendation['risk_metrics']['sharpe_ratio']}")
        print(f"  Bull/Base/Bear Returns: {recommendation['risk_metrics']['bull_case_return']}/{recommendation['risk_metrics']['base_case_return']}/{recommendation['risk_metrics']['bear_case_return']}")

    # Generate template
    print("\n" + "=" * 70)
    print("Financial Template Generated")
    print("=" * 70)
    print(generate_financial_template()[:500] + "...")
