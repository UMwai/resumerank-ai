"""
Patent Cliff Calendar Generator

Generates a forward-looking calendar of patent cliff events
with scoring and trade recommendations.
"""

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dateutil.relativedelta import relativedelta

from .scoring import CertaintyScoreCalculator, DrugPatentData, PatentCliffScorer
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CalendarEvent:
    """Data class for a patent cliff calendar event."""

    event_id: Optional[int]
    drug_id: int
    brand_name: str
    generic_name: str
    branded_company: str
    branded_company_ticker: Optional[str]
    event_type: str  # PATENT_EXPIRATION, ANDA_APPROVAL, COURT_DECISION, EXCLUSIVITY_END
    event_date: date
    related_patent_number: Optional[str]
    related_anda_number: Optional[str]
    certainty_score: float
    market_opportunity: int
    opportunity_tier: str
    trade_recommendation: str
    recommendation_confidence: str
    notes: Optional[str]
    days_until_event: int


class PatentCliffCalendarGenerator:
    """
    Generates patent cliff calendar from drug and patent data.

    The calendar provides a 12-month forward view of upcoming patent
    cliff events with certainty scores and trade recommendations.
    """

    EVENT_TYPES = {
        "PATENT_EXPIRATION": "Patent Expiration",
        "ANDA_APPROVAL": "Generic Approval Expected",
        "COURT_DECISION": "Court Decision Expected",
        "EXCLUSIVITY_END": "Exclusivity Period Ending",
    }

    def __init__(
        self,
        forward_months: int = 12,
        min_certainty: float = 0,
        min_opportunity: int = 0,
    ):
        """
        Initialize the calendar generator.

        Args:
            forward_months: Number of months to look ahead.
            min_certainty: Minimum certainty score to include.
            min_opportunity: Minimum market opportunity to include.
        """
        self.forward_months = forward_months
        self.min_certainty = min_certainty
        self.min_opportunity = min_opportunity
        self.scorer = PatentCliffScorer()

    def generate_from_database(self, db_loader) -> List[CalendarEvent]:
        """
        Generate calendar from database data.

        Args:
            db_loader: DatabaseLoader instance.

        Returns:
            List of CalendarEvent objects.
        """
        # Get drug/patent data from database
        drug_data = db_loader.get_drugs_for_calendar()

        events = []
        processed_drugs = set()

        today = date.today()
        end_date = today + relativedelta(months=self.forward_months)

        for row in drug_data:
            drug_id = row["drug_id"]

            # Skip if already processed
            if drug_id in processed_drugs:
                continue
            processed_drugs.add(drug_id)

            # Get expiration date
            exp_date = row.get("final_expiration_date")
            if not exp_date or not isinstance(exp_date, date):
                continue

            # Check if within forward window
            if exp_date < today or exp_date > end_date:
                continue

            # Build DrugPatentData for scoring
            drug_patent_data = DrugPatentData(
                drug_id=drug_id,
                brand_name=row.get("brand_name", ""),
                generic_name=row.get("generic_name", ""),
                branded_company=row.get("branded_company", ""),
                branded_company_ticker=row.get("branded_company_ticker"),
                annual_revenue=row.get("annual_revenue"),
                patent_numbers=[row.get("patent_number")] if row.get("patent_number") else [],
                earliest_expiration=exp_date,
                latest_expiration=exp_date,
                all_patents_expired=False,
                expiring_patents_count=1,
                total_patents_count=1,
                active_litigation_count=row.get("active_litigation", 0) or 0,
                resolved_litigation_count=0,
                patents_invalidated=0,
                approved_generics_count=row.get("approved_generics", 0) or 0,
                pending_generics_count=0,
                first_to_file_exists=False,
                pte_applied=False,
                pediatric_exclusivity=False,
            )

            # Score the event
            score_result = self.scorer.score_patent_cliff(drug_patent_data)

            certainty = score_result["scoring"]["final_certainty_score"]
            market_opp = score_result["market_opportunity"]["revenue_at_risk"]
            tier = score_result["market_opportunity"]["opportunity_tier"]

            # Apply filters
            if certainty < self.min_certainty:
                continue
            if market_opp < self.min_opportunity:
                continue

            # Create calendar event
            event = CalendarEvent(
                event_id=None,
                drug_id=drug_id,
                brand_name=row.get("brand_name", ""),
                generic_name=row.get("generic_name", ""),
                branded_company=row.get("branded_company", ""),
                branded_company_ticker=row.get("branded_company_ticker"),
                event_type="PATENT_EXPIRATION",
                event_date=exp_date,
                related_patent_number=row.get("patent_number"),
                related_anda_number=None,
                certainty_score=certainty,
                market_opportunity=market_opp,
                opportunity_tier=tier,
                trade_recommendation=score_result["trade_recommendation"]["recommendation"],
                recommendation_confidence=score_result["trade_recommendation"]["confidence"],
                notes=score_result["trade_recommendation"]["rationale"],
                days_until_event=(exp_date - today).days,
            )
            events.append(event)

        # Sort by event date
        events.sort(key=lambda x: x.event_date)

        logger.info(f"Generated {len(events)} calendar events for next {self.forward_months} months")
        return events

    def generate_from_raw_data(
        self, drugs: List[Dict], patents: List[Dict], andas: List[Dict]
    ) -> List[CalendarEvent]:
        """
        Generate calendar from raw extracted data.

        Args:
            drugs: List of drug dictionaries.
            patents: List of patent dictionaries.
            andas: List of ANDA dictionaries.

        Returns:
            List of CalendarEvent objects.
        """
        events = []
        today = date.today()
        end_date = today + relativedelta(months=self.forward_months)

        # Index patents by NDA number
        patents_by_nda = {}
        for patent in patents:
            nda = patent.get("nda_number")
            if nda not in patents_by_nda:
                patents_by_nda[nda] = []
            patents_by_nda[nda].append(patent)

        # Index ANDAs by active ingredient
        andas_by_ingredient = {}
        for anda in andas:
            ingredient = (anda.get("active_ingredient") or "").upper()
            if ingredient not in andas_by_ingredient:
                andas_by_ingredient[ingredient] = []
            andas_by_ingredient[ingredient].append(anda)

        for idx, drug in enumerate(drugs):
            nda_number = drug.get("nda_number")
            drug_patents = patents_by_nda.get(nda_number, [])

            if not drug_patents:
                continue

            # Find earliest expiration
            exp_dates = []
            for pat in drug_patents:
                exp_date = pat.get("base_expiration_date")
                if exp_date and isinstance(exp_date, date):
                    exp_dates.append(exp_date)

            if not exp_dates:
                continue

            earliest_exp = min(exp_dates)
            latest_exp = max(exp_dates)

            # Check if within window
            if earliest_exp < today or earliest_exp > end_date:
                continue

            # Get generic applications
            ingredient = (drug.get("generic_name") or drug.get("active_ingredient") or "").upper()
            drug_andas = andas_by_ingredient.get(ingredient, [])
            approved_generics = len([a for a in drug_andas if a.get("status") == "APPROVED"])

            # Build DrugPatentData
            drug_patent_data = DrugPatentData(
                drug_id=idx + 1,
                brand_name=drug.get("brand_name", ""),
                generic_name=drug.get("generic_name", ""),
                branded_company=drug.get("branded_company", ""),
                branded_company_ticker=drug.get("branded_company_ticker"),
                annual_revenue=drug.get("annual_revenue"),
                patent_numbers=[p.get("patent_number") for p in drug_patents if p.get("patent_number")],
                earliest_expiration=earliest_exp,
                latest_expiration=latest_exp,
                all_patents_expired=earliest_exp < today,
                expiring_patents_count=len([d for d in exp_dates if d <= end_date]),
                total_patents_count=len(drug_patents),
                active_litigation_count=0,
                resolved_litigation_count=0,
                patents_invalidated=0,
                approved_generics_count=approved_generics,
                pending_generics_count=0,
                first_to_file_exists=False,
                pte_applied=False,
                pediatric_exclusivity=False,
            )

            # Score the event
            score_result = self.scorer.score_patent_cliff(drug_patent_data)

            certainty = score_result["scoring"]["final_certainty_score"]
            market_opp = score_result["market_opportunity"]["revenue_at_risk"]
            tier = score_result["market_opportunity"]["opportunity_tier"]

            # Apply filters
            if certainty < self.min_certainty:
                continue
            if market_opp < self.min_opportunity:
                continue

            # Create calendar event
            event = CalendarEvent(
                event_id=None,
                drug_id=idx + 1,
                brand_name=drug.get("brand_name", ""),
                generic_name=drug.get("generic_name", ""),
                branded_company=drug.get("branded_company", ""),
                branded_company_ticker=drug.get("branded_company_ticker"),
                event_type="PATENT_EXPIRATION",
                event_date=earliest_exp,
                related_patent_number=drug_patents[0].get("patent_number") if drug_patents else None,
                related_anda_number=None,
                certainty_score=certainty,
                market_opportunity=market_opp,
                opportunity_tier=tier,
                trade_recommendation=score_result["trade_recommendation"]["recommendation"],
                recommendation_confidence=score_result["trade_recommendation"]["confidence"],
                notes=score_result["trade_recommendation"]["rationale"],
                days_until_event=(earliest_exp - today).days,
            )
            events.append(event)

        # Sort by event date
        events.sort(key=lambda x: x.event_date)

        logger.info(f"Generated {len(events)} calendar events from raw data")
        return events

    def to_dataframe(self, events: List[CalendarEvent]):
        """
        Convert events to pandas DataFrame.

        Args:
            events: List of CalendarEvent objects.

        Returns:
            pandas DataFrame.
        """
        import pandas as pd

        records = []
        for event in events:
            records.append({
                "Event Date": event.event_date,
                "Brand Name": event.brand_name,
                "Generic Name": event.generic_name,
                "Company": event.branded_company,
                "Ticker": event.branded_company_ticker,
                "Event Type": self.EVENT_TYPES.get(event.event_type, event.event_type),
                "Patent Number": event.related_patent_number,
                "Certainty Score": f"{event.certainty_score:.1f}%",
                "Market Opportunity": f"${event.market_opportunity:,.0f}",
                "Opportunity Tier": event.opportunity_tier,
                "Recommendation": event.trade_recommendation,
                "Confidence": event.recommendation_confidence,
                "Days Until Event": event.days_until_event,
                "Notes": event.notes,
            })

        return pd.DataFrame(records)

    def export_csv(self, events: List[CalendarEvent], filepath: str) -> None:
        """
        Export calendar to CSV file.

        Args:
            events: List of CalendarEvent objects.
            filepath: Output file path.
        """
        df = self.to_dataframe(events)
        df.to_csv(filepath, index=False)
        logger.info(f"Calendar exported to {filepath}")

    def export_json(self, events: List[CalendarEvent], filepath: str) -> None:
        """
        Export calendar to JSON file.

        Args:
            events: List of CalendarEvent objects.
            filepath: Output file path.
        """
        records = []
        for event in events:
            records.append({
                "event_id": event.event_id,
                "drug_id": event.drug_id,
                "brand_name": event.brand_name,
                "generic_name": event.generic_name,
                "branded_company": event.branded_company,
                "branded_company_ticker": event.branded_company_ticker,
                "event_type": event.event_type,
                "event_date": event.event_date.isoformat(),
                "related_patent_number": event.related_patent_number,
                "related_anda_number": event.related_anda_number,
                "certainty_score": event.certainty_score,
                "market_opportunity": event.market_opportunity,
                "opportunity_tier": event.opportunity_tier,
                "trade_recommendation": event.trade_recommendation,
                "recommendation_confidence": event.recommendation_confidence,
                "notes": event.notes,
                "days_until_event": event.days_until_event,
            })

        with open(filepath, "w") as f:
            json.dump(records, f, indent=2)

        logger.info(f"Calendar exported to {filepath}")

    def format_calendar_report(self, events: List[CalendarEvent]) -> str:
        """
        Format calendar as a readable text report.

        Args:
            events: List of CalendarEvent objects.

        Returns:
            Formatted report string.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PATENT CLIFF CALENDAR - 12 MONTH FORWARD VIEW")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        total_events = len(events)
        high_conf = len([e for e in events if e.recommendation_confidence == "HIGH"])
        medium_conf = len([e for e in events if e.recommendation_confidence == "MEDIUM"])
        blockbuster = len([e for e in events if e.opportunity_tier == "BLOCKBUSTER"])
        total_opportunity = sum(e.market_opportunity for e in events)

        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Events: {total_events}")
        lines.append(f"High Confidence: {high_conf}")
        lines.append(f"Medium Confidence: {medium_conf}")
        lines.append(f"Blockbuster Opportunities: {blockbuster}")
        lines.append(f"Total Market at Risk: ${total_opportunity:,.0f}")
        lines.append("")

        # Events by month
        current_month = None
        for event in events:
            event_month = event.event_date.strftime("%B %Y")

            if event_month != current_month:
                lines.append("")
                lines.append(f"--- {event_month} ---")
                current_month = event_month

            lines.append("")
            lines.append(f"[{event.event_date.strftime('%Y-%m-%d')}] {event.brand_name}")
            lines.append(f"  Generic: {event.generic_name}")
            lines.append(f"  Company: {event.branded_company} ({event.branded_company_ticker or 'N/A'})")
            lines.append(f"  Event: {self.EVENT_TYPES.get(event.event_type, event.event_type)}")
            if event.related_patent_number:
                lines.append(f"  Patent: {event.related_patent_number}")
            lines.append(f"  Certainty: {event.certainty_score:.1f}%")
            lines.append(f"  Market Opportunity: ${event.market_opportunity:,.0f} ({event.opportunity_tier})")
            lines.append(f"  Recommendation: {event.trade_recommendation} [{event.recommendation_confidence}]")
            if event.notes:
                lines.append(f"  Notes: {event.notes}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def get_events_for_database(self, events: List[CalendarEvent]) -> List[Dict[str, Any]]:
        """
        Convert events to database-ready dictionaries.

        Args:
            events: List of CalendarEvent objects.

        Returns:
            List of dictionaries for database insertion.
        """
        records = []
        for event in events:
            records.append({
                "drug_id": event.drug_id,
                "event_type": event.event_type,
                "event_date": event.event_date,
                "related_patent_number": event.related_patent_number,
                "related_anda_number": event.related_anda_number,
                "certainty_score": event.certainty_score,
                "market_opportunity": event.market_opportunity,
                "opportunity_tier": event.opportunity_tier,
                "trade_recommendation": event.trade_recommendation,
                "recommendation_confidence": event.recommendation_confidence,
                "notes": event.notes,
            })
        return records


if __name__ == "__main__":
    # Test the calendar generator with sample data
    print("\n=== Testing Patent Cliff Calendar Generator ===")

    # Sample drugs
    sample_drugs = [
        {
            "nda_number": "NDA001",
            "brand_name": "DrugA",
            "generic_name": "generica",
            "active_ingredient": "generica",
            "branded_company": "PharmaCo Inc.",
            "branded_company_ticker": "PHRM",
            "annual_revenue": 5_000_000_000,
        },
        {
            "nda_number": "NDA002",
            "brand_name": "DrugB",
            "generic_name": "genericb",
            "active_ingredient": "genericb",
            "branded_company": "BioPharma Ltd.",
            "branded_company_ticker": "BIOP",
            "annual_revenue": 2_000_000_000,
        },
    ]

    # Sample patents
    sample_patents = [
        {
            "nda_number": "NDA001",
            "patent_number": "US1234567",
            "base_expiration_date": date(2025, 6, 15),
        },
        {
            "nda_number": "NDA002",
            "patent_number": "US7654321",
            "base_expiration_date": date(2025, 3, 30),
        },
    ]

    # Sample ANDAs
    sample_andas = [
        {
            "anda_number": "ANDA001",
            "active_ingredient": "GENERICA",
            "status": "APPROVED",
        },
        {
            "anda_number": "ANDA002",
            "active_ingredient": "GENERICA",
            "status": "APPROVED",
        },
    ]

    generator = PatentCliffCalendarGenerator(forward_months=12)
    events = generator.generate_from_raw_data(sample_drugs, sample_patents, sample_andas)

    print(f"\nGenerated {len(events)} events")

    # Print formatted report
    report = generator.format_calendar_report(events)
    print(report)
