"""
Tests for Financial Modeling module.
"""

import pytest
from datetime import date

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.financial_models import (
    RevenueErosionModel,
    ErosionForecast,
    GenericMarketModel,
    NPVCalculator,
    NPVAnalysis,
    TradeRecommendationEngine,
    generate_financial_template,
)


class TestRevenueErosionModel:
    """Tests for RevenueErosionModel class."""

    def test_init(self):
        """Test initialization."""
        model = RevenueErosionModel()
        assert model is not None

    def test_determine_drug_type_biologic(self):
        """Test drug type determination for biologics."""
        model = RevenueErosionModel()

        drug_type = model.determine_drug_type(
            is_biologic=True,
            is_complex=False,
            num_competitors=3,
        )
        assert drug_type == "biologic"

    def test_determine_drug_type_complex(self):
        """Test drug type determination for complex generics."""
        model = RevenueErosionModel()

        drug_type = model.determine_drug_type(
            is_biologic=False,
            is_complex=True,
            num_competitors=3,
        )
        assert drug_type == "complex_generic"

    def test_determine_drug_type_small_molecule(self):
        """Test drug type determination for small molecules."""
        model = RevenueErosionModel()

        drug_type = model.determine_drug_type(
            is_biologic=False,
            is_complex=False,
            num_competitors=5,
        )
        assert drug_type == "small_molecule_multiple"

    def test_forecast_erosion_basic(self):
        """Test basic erosion forecast."""
        model = RevenueErosionModel()

        forecast = model.forecast_erosion(
            drug_name="TestDrug",
            brand_company="TestCompany",
            base_revenue=10000000000,
            therapeutic_area="Oncology",
            is_biologic=False,
            num_generic_competitors=3,
        )

        assert isinstance(forecast, ErosionForecast)
        assert forecast.drug_name == "TestDrug"
        assert forecast.base_revenue == 10000000000
        assert forecast.year_1_revenue < forecast.base_revenue
        assert forecast.year_5_revenue < forecast.year_1_revenue

    def test_forecast_erosion_biologic(self):
        """Test erosion forecast for biologics (slower erosion)."""
        model = RevenueErosionModel()

        biologic_forecast = model.forecast_erosion(
            drug_name="BiologicDrug",
            brand_company="Company",
            base_revenue=10000000000,
            is_biologic=True,
            num_generic_competitors=3,
        )

        small_molecule_forecast = model.forecast_erosion(
            drug_name="SmallMolecule",
            brand_company="Company",
            base_revenue=10000000000,
            is_biologic=False,
            num_generic_competitors=3,
        )

        # Biologics should have slower erosion
        assert biologic_forecast.year_1_erosion_pct < small_molecule_forecast.year_1_erosion_pct

    def test_forecast_scenarios(self):
        """Test multi-scenario forecasting."""
        model = RevenueErosionModel()

        forecasts = model.multi_scenario_forecast(
            drug_name="TestDrug",
            brand_company="Company",
            base_revenue=10000000000,
        )

        assert "conservative" in forecasts
        assert "base" in forecasts
        assert "aggressive" in forecasts

        # Aggressive should show more erosion than conservative
        assert forecasts["aggressive"].year_1_erosion_pct > forecasts["conservative"].year_1_erosion_pct

    def test_forecast_to_dict(self):
        """Test forecast serialization."""
        model = RevenueErosionModel()

        forecast = model.forecast_erosion(
            drug_name="TestDrug",
            brand_company="Company",
            base_revenue=10000000000,
        )

        forecast_dict = forecast.to_dict()
        assert isinstance(forecast_dict, dict)
        assert "drug_name" in forecast_dict
        assert "year_1_revenue" in forecast_dict


class TestGenericMarketModel:
    """Tests for GenericMarketModel class."""

    def test_project_market_share_early(self):
        """Test market share projection early after entry."""
        model = GenericMarketModel()

        projection = model.project_generic_market_share(
            base_revenue=10000000000,
            months_since_generic_entry=6,
            num_competitors=2,
        )

        assert "generic_market_capture_pct" in projection
        assert "branded_market_share_pct" in projection
        assert projection["generic_market_capture_pct"] > 0
        assert projection["branded_market_share_pct"] > 0

    def test_project_market_share_mature(self):
        """Test market share projection at maturity."""
        model = GenericMarketModel()

        projection = model.project_generic_market_share(
            base_revenue=10000000000,
            months_since_generic_entry=36,
            num_competitors=5,
        )

        # Mature market should have high generic capture
        assert projection["generic_market_capture_pct"] > 50

    def test_biologic_slower_uptake(self):
        """Test that biologics have slower generic uptake."""
        model = GenericMarketModel()

        biologic_projection = model.project_generic_market_share(
            base_revenue=10000000000,
            months_since_generic_entry=12,
            num_competitors=3,
            is_biologic=True,
        )

        small_molecule_projection = model.project_generic_market_share(
            base_revenue=10000000000,
            months_since_generic_entry=12,
            num_competitors=3,
            is_biologic=False,
        )

        assert biologic_projection["generic_market_capture_pct"] < small_molecule_projection["generic_market_capture_pct"]


class TestNPVCalculator:
    """Tests for NPVCalculator class."""

    def test_init_default(self):
        """Test default initialization."""
        calc = NPVCalculator()
        assert calc.discount_rate == 0.10

    def test_init_custom_rate(self):
        """Test custom discount rate."""
        calc = NPVCalculator(discount_rate=0.08)
        assert calc.discount_rate == 0.08

    def test_calculate_npv_basic(self):
        """Test basic NPV calculation."""
        calc = NPVCalculator(discount_rate=0.10)

        # Simple cash flows
        cash_flows = [1000, 1000, 1000]
        npv = calc.calculate_npv(cash_flows)

        # NPV should be less than sum due to discounting
        assert npv < sum(cash_flows)
        assert npv > 0

    def test_calculate_risk_adjusted_npv(self):
        """Test risk-adjusted NPV calculation."""
        calc = NPVCalculator()

        # Create mock erosion forecast
        erosion = ErosionForecast(
            drug_name="TestDrug",
            brand_company="Company",
            base_revenue=10000000000,
            year_1_revenue=6000000000,
            year_2_revenue=4000000000,
            year_3_revenue=3000000000,
            year_4_revenue=2500000000,
            year_5_revenue=2000000000,
            year_1_erosion_pct=40,
            year_2_erosion_pct=60,
            year_3_erosion_pct=70,
            year_4_erosion_pct=75,
            year_5_erosion_pct=80,
            total_revenue_loss_5yr=32500000000,
            cumulative_erosion_pct=80,
            generic_market_year_1=1200000000,
            generic_market_year_5=1600000000,
            num_generic_competitors_projected=5,
            forecast_confidence="medium",
            scenario="base",
        )

        analysis = calc.calculate_risk_adjusted_npv(
            drug_name="TestDrug",
            erosion_forecast=erosion,
            upheld_probability=40.0,
            portfolio_value=1000000,
        )

        assert isinstance(analysis, NPVAnalysis)
        assert analysis.npv_branded_decline > 0
        assert analysis.recommended_position_size_pct > 0
        assert analysis.win_probability > 0


class TestTradeRecommendationEngine:
    """Tests for TradeRecommendationEngine class."""

    def test_generate_recommendation_execute(self):
        """Test recommendation for high-probability trade."""
        engine = TradeRecommendationEngine()

        rec = engine.generate_recommendation(
            drug_name="TestDrug",
            brand_company="Company",
            brand_ticker="TEST",
            base_revenue=10000000000,
            therapeutic_area="Oncology",
            is_biologic=False,
            num_competitors=5,
            upheld_probability=30.0,  # Low = good for short
            days_until_event=90,
        )

        assert rec["action"] in ["EXECUTE", "INITIATE"]
        assert "trade_structure" in rec

    def test_generate_recommendation_avoid(self):
        """Test recommendation for low-probability trade."""
        engine = TradeRecommendationEngine()

        rec = engine.generate_recommendation(
            drug_name="TestDrug",
            brand_company="Company",
            brand_ticker="TEST",
            base_revenue=10000000000,
            therapeutic_area="Oncology",
            is_biologic=True,
            num_competitors=2,
            upheld_probability=80.0,  # High = bad for short
            days_until_event=90,
        )

        assert rec["action"] in ["AVOID", "MONITOR"]

    def test_recommendation_structure(self):
        """Test recommendation includes all required fields."""
        engine = TradeRecommendationEngine()

        rec = engine.generate_recommendation(
            drug_name="TestDrug",
            brand_company="Company",
            brand_ticker="TEST",
            base_revenue=10000000000,
            therapeutic_area="Cardiovascular",
            is_biologic=False,
            num_competitors=4,
            upheld_probability=50.0,
            days_until_event=180,
        )

        required_fields = [
            "drug_name",
            "action",
            "action_rationale",
            "financials",
            "npv_analysis",
            "position_sizing",
            "risk_metrics",
        ]

        for field in required_fields:
            assert field in rec


class TestFinancialTemplate:
    """Tests for financial template generation."""

    def test_generate_template(self):
        """Test template generation."""
        template = generate_financial_template()

        assert isinstance(template, str)
        assert "INPUTS" in template
        assert "NPV ANALYSIS" in template
        assert "POSITION SIZING" in template
