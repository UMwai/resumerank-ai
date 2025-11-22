"""
Data fetchers for each intelligence system.

These classes encapsulate the database queries for each system
and provide standardized data structures for the dashboard.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class SignalOpportunity:
    """Unified data structure for opportunities across all systems."""
    ticker: str
    company_name: str
    signal_type: str
    source_system: str  # 'clinical_trials', 'patent_intelligence', 'insider_hiring'
    score: float
    confidence: float
    recommendation: str
    description: str
    event_date: Optional[date] = None
    details: Dict[str, Any] = field(default_factory=dict)


class ClinicalTrialFetcher:
    """Fetches data from the Clinical Trial Signals database."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.db_name = 'clinical_trials'

    def get_recent_signals(self, days: int = 7) -> pd.DataFrame:
        """
        Get recent trial signals.

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with signal data
        """
        query = """
            SELECT
                ts.signal_id,
                ts.trial_id,
                ts.signal_type,
                ts.signal_value,
                ts.signal_weight,
                ts.detected_date,
                ts.source,
                t.drug_name,
                t.indication,
                t.phase,
                t.company_ticker,
                c.company_name
            FROM trial_signals ts
            JOIN trials t ON ts.trial_id = t.trial_id
            LEFT JOIN companies c ON t.company_ticker = c.ticker
            WHERE ts.detected_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY ts.detected_date DESC
        """
        results = self.db.execute_query(self.db_name, query, (days,))
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_upcoming_readouts(self, days: int = 90) -> pd.DataFrame:
        """
        Get trials with upcoming primary completion dates.

        Args:
            days: Number of days to look forward

        Returns:
            DataFrame with upcoming readouts
        """
        query = """
            SELECT
                t.trial_id,
                t.drug_name,
                t.indication,
                t.phase,
                t.primary_completion_date,
                t.expected_completion,
                t.enrollment_current,
                t.enrollment_target,
                t.status,
                t.company_ticker,
                c.company_name,
                c.market_cap,
                ts.composite_score,
                ts.confidence,
                ts.recommendation
            FROM trials t
            LEFT JOIN companies c ON t.company_ticker = c.ticker
            LEFT JOIN (
                SELECT DISTINCT ON (trial_id) *
                FROM trial_scores
                ORDER BY trial_id, score_date DESC
            ) ts ON t.trial_id = ts.trial_id
            WHERE t.primary_completion_date IS NOT NULL
            AND t.primary_completion_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '%s days'
            AND t.status NOT IN ('COMPLETED', 'TERMINATED', 'WITHDRAWN')
            ORDER BY t.primary_completion_date ASC
        """
        results = self.db.execute_query(self.db_name, query, (days,))
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_top_opportunities(self, limit: int = 10) -> pd.DataFrame:
        """
        Get top actionable opportunities based on scores.

        Args:
            limit: Maximum number of opportunities

        Returns:
            DataFrame with top opportunities
        """
        query = """
            SELECT
                ts.trial_id,
                ts.composite_score,
                ts.confidence,
                ts.recommendation,
                ts.score_date,
                ts.contributing_signals,
                t.drug_name,
                t.indication,
                t.phase,
                t.primary_completion_date,
                t.company_ticker,
                c.company_name,
                c.market_cap
            FROM trial_scores ts
            JOIN trials t ON ts.trial_id = t.trial_id
            LEFT JOIN companies c ON t.company_ticker = c.ticker
            WHERE ts.score_date >= CURRENT_DATE - INTERVAL '7 days'
            AND (ts.composite_score >= 7 OR ts.composite_score <= 3)
            ORDER BY ABS(ts.composite_score - 5) DESC, ts.confidence DESC
            LIMIT %s
        """
        results = self.db.execute_query(self.db_name, query, (limit,))
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_monitored_trials(self) -> pd.DataFrame:
        """
        Get all monitored (active) trials.

        Returns:
            DataFrame with monitored trials
        """
        query = """
            SELECT
                t.*,
                c.company_name,
                c.market_cap,
                ts.composite_score,
                ts.confidence,
                ts.recommendation
            FROM trials t
            LEFT JOIN companies c ON t.company_ticker = c.ticker
            LEFT JOIN (
                SELECT DISTINCT ON (trial_id) *
                FROM trial_scores
                ORDER BY trial_id, score_date DESC
            ) ts ON t.trial_id = ts.trial_id
            WHERE t.status NOT IN ('COMPLETED', 'TERMINATED', 'WITHDRAWN')
            ORDER BY t.expected_completion ASC NULLS LAST
        """
        results = self.db.execute_query(self.db_name, query)
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_companies(self) -> pd.DataFrame:
        """Get all companies being monitored."""
        query = "SELECT * FROM companies ORDER BY ticker"
        results = self.db.execute_query(self.db_name, query)
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_signal_opportunities(self) -> List[SignalOpportunity]:
        """Convert top opportunities to unified SignalOpportunity format."""
        df = self.get_top_opportunities(20)
        opportunities = []

        for _, row in df.iterrows():
            opportunities.append(SignalOpportunity(
                ticker=row.get('company_ticker', ''),
                company_name=row.get('company_name', ''),
                signal_type='clinical_trial',
                source_system='clinical_trials',
                score=float(row.get('composite_score', 5)) / 10,  # Normalize to 0-1
                confidence=float(row.get('confidence', 0)),
                recommendation=row.get('recommendation', 'HOLD'),
                description=f"{row.get('drug_name', 'N/A')} - {row.get('indication', 'N/A')} (Phase {row.get('phase', 'N/A')})",
                event_date=row.get('primary_completion_date'),
                details={
                    'trial_id': row.get('trial_id'),
                    'drug_name': row.get('drug_name'),
                    'indication': row.get('indication'),
                    'phase': row.get('phase'),
                }
            ))

        return opportunities


class PatentIntelligenceFetcher:
    """Fetches data from the Patent Intelligence database."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.db_name = 'patent_intelligence'

    def get_patent_calendar(self, months: int = 12) -> pd.DataFrame:
        """
        Get patent expiration calendar for the next N months.

        Args:
            months: Number of months to look forward

        Returns:
            DataFrame with patent expirations
        """
        query = """
            SELECT
                d.drug_id,
                d.brand_name,
                d.generic_name,
                d.branded_company,
                d.branded_company_ticker,
                d.annual_revenue,
                p.patent_number,
                p.expiration_date,
                p.patent_type,
                s.certainty_score,
                s.recommendation,
                s.market_opportunity,
                s.revenue_at_risk
            FROM drugs d
            JOIN drug_patents dp ON d.drug_id = dp.drug_id
            JOIN patents p ON dp.patent_id = p.patent_id
            LEFT JOIN patent_cliff_scores s ON d.drug_id = s.drug_id
            WHERE p.expiration_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '%s months'
            ORDER BY p.expiration_date ASC
        """
        results = self.db.execute_query(self.db_name, query, (months,))
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_generic_opportunities(self, min_revenue: int = 100_000_000) -> pd.DataFrame:
        """
        Get high-value generic entry opportunities.

        Args:
            min_revenue: Minimum annual revenue to consider

        Returns:
            DataFrame with generic opportunities
        """
        query = """
            SELECT
                d.drug_id,
                d.brand_name,
                d.generic_name,
                d.branded_company,
                d.branded_company_ticker,
                d.annual_revenue,
                s.certainty_score,
                s.recommendation,
                s.confidence,
                s.market_opportunity,
                s.revenue_at_risk,
                s.trade_recommendation,
                COUNT(DISTINCT a.anda_id) as anda_count,
                MIN(p.expiration_date) as earliest_expiration
            FROM drugs d
            LEFT JOIN patent_cliff_scores s ON d.drug_id = s.drug_id
            LEFT JOIN anda_filings a ON d.drug_id = a.drug_id
            LEFT JOIN drug_patents dp ON d.drug_id = dp.drug_id
            LEFT JOIN patents p ON dp.patent_id = p.patent_id
            WHERE d.annual_revenue >= %s
            GROUP BY d.drug_id, d.brand_name, d.generic_name, d.branded_company,
                     d.branded_company_ticker, d.annual_revenue, s.certainty_score,
                     s.recommendation, s.confidence, s.market_opportunity,
                     s.revenue_at_risk, s.trade_recommendation
            ORDER BY s.certainty_score DESC NULLS LAST, d.annual_revenue DESC
        """
        results = self.db.execute_query(self.db_name, query, (min_revenue,))
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_trade_recommendations(self) -> pd.DataFrame:
        """
        Get active trade recommendations.

        Returns:
            DataFrame with trade recommendations
        """
        query = """
            SELECT
                s.score_id,
                s.drug_id,
                d.brand_name,
                d.generic_name,
                d.branded_company,
                d.branded_company_ticker,
                d.annual_revenue,
                s.certainty_score,
                s.recommendation,
                s.confidence,
                s.market_opportunity,
                s.revenue_at_risk,
                s.trade_recommendation,
                s.trade_type,
                s.days_until_event,
                s.score_date
            FROM patent_cliff_scores s
            JOIN drugs d ON s.drug_id = d.drug_id
            WHERE s.recommendation IN ('EXECUTE TRADE', 'INITIATE POSITION')
            AND s.score_date >= CURRENT_DATE - INTERVAL '30 days'
            ORDER BY s.certainty_score DESC, s.revenue_at_risk DESC
        """
        results = self.db.execute_query(self.db_name, query)
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_litigation_status(self) -> pd.DataFrame:
        """
        Get current litigation status for drugs.

        Returns:
            DataFrame with litigation data
        """
        query = """
            SELECT
                d.drug_id,
                d.brand_name,
                d.branded_company,
                d.branded_company_ticker,
                l.litigation_id,
                l.case_name,
                l.status,
                l.filing_date,
                l.generic_company,
                l.outcome
            FROM drugs d
            JOIN litigation l ON d.drug_id = l.drug_id
            WHERE l.status = 'ACTIVE'
            ORDER BY l.filing_date DESC
        """
        results = self.db.execute_query(self.db_name, query)
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_signal_opportunities(self) -> List[SignalOpportunity]:
        """Convert top opportunities to unified SignalOpportunity format."""
        df = self.get_generic_opportunities()
        opportunities = []

        for _, row in df.iterrows():
            score = float(row.get('certainty_score', 0)) / 100 if row.get('certainty_score') else 0
            opportunities.append(SignalOpportunity(
                ticker=row.get('branded_company_ticker', ''),
                company_name=row.get('branded_company', ''),
                signal_type='patent_cliff',
                source_system='patent_intelligence',
                score=score,
                confidence=float(row.get('confidence', 0)) / 100 if row.get('confidence') else score,
                recommendation=row.get('recommendation', 'MONITOR'),
                description=f"{row.get('brand_name', 'N/A')} ({row.get('generic_name', 'N/A')}) - ${row.get('annual_revenue', 0):,.0f} revenue",
                event_date=row.get('earliest_expiration'),
                details={
                    'drug_id': row.get('drug_id'),
                    'brand_name': row.get('brand_name'),
                    'generic_name': row.get('generic_name'),
                    'annual_revenue': row.get('annual_revenue'),
                    'revenue_at_risk': row.get('revenue_at_risk'),
                    'anda_count': row.get('anda_count'),
                }
            ))

        return opportunities


class InsiderHiringFetcher:
    """Fetches data from the Insider Activity + Hiring Signals database."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.db_name = 'insider_hiring'

    def get_recent_form4(self, days: int = 30) -> pd.DataFrame:
        """
        Get recent Form 4 filings (insider transactions).

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with Form 4 data
        """
        query = """
            SELECT
                it.transaction_id,
                it.company_ticker,
                c.company_name,
                it.insider_name,
                it.insider_title,
                it.transaction_date,
                it.transaction_type,
                it.shares,
                it.price_per_share,
                it.transaction_value,
                it.shares_owned_after,
                it.is_10b5_1_plan,
                it.is_director,
                it.is_officer,
                it.signal_weight
            FROM insider_transactions it
            LEFT JOIN companies c ON it.company_ticker = c.ticker
            WHERE it.transaction_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY it.transaction_date DESC, it.transaction_value DESC
        """
        results = self.db.execute_query(self.db_name, query, (days,))
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_institutional_changes(self, quarter: Optional[str] = None) -> pd.DataFrame:
        """
        Get recent 13F institutional holding changes.

        Args:
            quarter: Optional quarter filter (e.g., '2024Q3')

        Returns:
            DataFrame with 13F changes
        """
        query = """
            SELECT
                ih.holding_id,
                ih.fund_name,
                ih.company_ticker,
                c.company_name,
                ih.quarter_end,
                ih.shares,
                ih.market_value,
                ih.pct_change_shares,
                ih.pct_portfolio,
                ih.is_new_position,
                ih.is_exit,
                ih.signal_weight
            FROM institutional_holdings ih
            LEFT JOIN companies c ON ih.company_ticker = c.ticker
            WHERE ih.quarter_end >= CURRENT_DATE - INTERVAL '180 days'
            ORDER BY ih.quarter_end DESC, ABS(ih.pct_change_shares) DESC NULLS LAST
        """
        results = self.db.execute_query(self.db_name, query)
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_job_posting_trends(self, days: int = 90) -> pd.DataFrame:
        """
        Get job posting trends by company.

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with job posting trends
        """
        query = """
            SELECT
                jp.company_ticker,
                c.company_name,
                COUNT(*) as total_jobs,
                SUM(CASE WHEN jp.is_commercial_role THEN 1 ELSE 0 END) as commercial_jobs,
                SUM(CASE WHEN jp.is_manufacturing_role THEN 1 ELSE 0 END) as manufacturing_jobs,
                SUM(CASE WHEN jp.is_clinical_role THEN 1 ELSE 0 END) as clinical_jobs,
                SUM(CASE WHEN jp.is_rd_role THEN 1 ELSE 0 END) as rd_jobs,
                SUM(CASE WHEN jp.is_senior_role THEN 1 ELSE 0 END) as senior_jobs,
                SUM(CASE WHEN jp.removal_date IS NOT NULL THEN 1 ELSE 0 END) as removed_jobs
            FROM job_postings jp
            LEFT JOIN companies c ON jp.company_ticker = c.ticker
            WHERE jp.first_seen_date >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY jp.company_ticker, c.company_name
            ORDER BY total_jobs DESC
        """
        results = self.db.execute_query(self.db_name, query, (days,))
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_signal_scores(self, limit: int = 50) -> pd.DataFrame:
        """
        Get latest signal scores for companies.

        Args:
            limit: Maximum number of scores

        Returns:
            DataFrame with signal scores
        """
        query = """
            SELECT
                ss.score_id,
                ss.company_ticker,
                c.company_name,
                ss.score_date,
                ss.composite_score,
                ss.confidence,
                ss.signal_count,
                ss.insider_score,
                ss.institutional_score,
                ss.hiring_score,
                ss.sentiment_score,
                ss.recommendation,
                ss.contributing_signals
            FROM signal_scores ss
            LEFT JOIN companies c ON ss.company_ticker = c.ticker
            WHERE ss.score_date >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY ABS(ss.composite_score) DESC, ss.confidence DESC
            LIMIT %s
        """
        results = self.db.execute_query(self.db_name, query, (limit,))
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_top_insider_buys(self, days: int = 30, limit: int = 10) -> pd.DataFrame:
        """
        Get top insider purchases by value.

        Args:
            days: Number of days to look back
            limit: Maximum number of results

        Returns:
            DataFrame with top insider buys
        """
        query = """
            SELECT
                it.company_ticker,
                c.company_name,
                SUM(it.transaction_value) as total_value,
                COUNT(*) as transaction_count,
                STRING_AGG(DISTINCT it.insider_name, ', ') as insiders
            FROM insider_transactions it
            LEFT JOIN companies c ON it.company_ticker = c.ticker
            WHERE it.transaction_date >= CURRENT_DATE - INTERVAL '%s days'
            AND it.transaction_type = 'Purchase'
            AND it.is_10b5_1_plan = FALSE
            GROUP BY it.company_ticker, c.company_name
            ORDER BY total_value DESC
            LIMIT %s
        """
        results = self.db.execute_query(self.db_name, query, (days, limit))
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_signal_opportunities(self) -> List[SignalOpportunity]:
        """Convert signal scores to unified SignalOpportunity format."""
        df = self.get_signal_scores()
        opportunities = []

        for _, row in df.iterrows():
            score = (float(row.get('composite_score', 0)) + 10) / 20  # Normalize -10 to +10 to 0-1
            opportunities.append(SignalOpportunity(
                ticker=row.get('company_ticker', ''),
                company_name=row.get('company_name', ''),
                signal_type='insider_hiring',
                source_system='insider_hiring',
                score=score,
                confidence=float(row.get('confidence', 0)),
                recommendation=row.get('recommendation', 'NEUTRAL'),
                description=f"Insider: {row.get('insider_score', 0):.1f} | Inst: {row.get('institutional_score', 0):.1f} | Hiring: {row.get('hiring_score', 0):.1f}",
                event_date=row.get('score_date'),
                details={
                    'signal_count': row.get('signal_count'),
                    'insider_score': row.get('insider_score'),
                    'institutional_score': row.get('institutional_score'),
                    'hiring_score': row.get('hiring_score'),
                }
            ))

        return opportunities


class CombinedSignalFetcher:
    """
    Combines signals from all three systems to generate unified scores.

    Uses weighted scoring to combine:
    - Clinical Trial signals (40%)
    - Patent Intelligence signals (30%)
    - Insider/Hiring signals (30%)
    """

    # Weights for combining signals
    WEIGHTS = {
        'clinical_trials': 0.40,
        'patent_intelligence': 0.30,
        'insider_hiring': 0.30,
    }

    def __init__(
        self,
        clinical_fetcher: ClinicalTrialFetcher,
        patent_fetcher: PatentIntelligenceFetcher,
        insider_fetcher: InsiderHiringFetcher,
    ):
        self.clinical = clinical_fetcher
        self.patent = patent_fetcher
        self.insider = insider_fetcher

    def get_all_opportunities(self) -> List[SignalOpportunity]:
        """
        Get all signal opportunities from all systems.

        Returns:
            List of SignalOpportunity objects
        """
        opportunities = []

        try:
            opportunities.extend(self.clinical.get_signal_opportunities())
        except Exception as e:
            logger.warning(f"Failed to get clinical trial opportunities: {e}")

        try:
            opportunities.extend(self.patent.get_signal_opportunities())
        except Exception as e:
            logger.warning(f"Failed to get patent opportunities: {e}")

        try:
            opportunities.extend(self.insider.get_signal_opportunities())
        except Exception as e:
            logger.warning(f"Failed to get insider/hiring opportunities: {e}")

        return opportunities

    def calculate_combined_score(
        self,
        ticker: str,
        opportunities: Optional[List[SignalOpportunity]] = None
    ) -> Dict[str, Any]:
        """
        Calculate combined confidence score for a ticker.

        Args:
            ticker: Company ticker symbol
            opportunities: Optional pre-fetched opportunities

        Returns:
            Dictionary with combined score and breakdown
        """
        if opportunities is None:
            opportunities = self.get_all_opportunities()

        # Filter opportunities for this ticker
        ticker_opps = [o for o in opportunities if o.ticker == ticker]

        if not ticker_opps:
            return {
                'ticker': ticker,
                'combined_score': 0,
                'confidence': 0,
                'signal_count': 0,
                'breakdown': {},
                'recommendation': 'NO DATA',
            }

        # Calculate weighted score by system
        system_scores = {}
        for opp in ticker_opps:
            system = opp.source_system
            if system not in system_scores:
                system_scores[system] = {'scores': [], 'confidences': []}
            system_scores[system]['scores'].append(opp.score)
            system_scores[system]['confidences'].append(opp.confidence)

        # Average scores per system
        breakdown = {}
        weighted_score = 0
        weighted_confidence = 0
        total_weight = 0

        for system, data in system_scores.items():
            avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
            avg_conf = sum(data['confidences']) / len(data['confidences']) if data['confidences'] else 0
            weight = self.WEIGHTS.get(system, 0)

            breakdown[system] = {
                'score': avg_score,
                'confidence': avg_conf,
                'weight': weight,
                'signal_count': len(data['scores']),
            }

            weighted_score += avg_score * weight
            weighted_confidence += avg_conf * weight
            total_weight += weight

        # Normalize if we don't have all systems
        if total_weight > 0:
            weighted_score /= total_weight
            weighted_confidence /= total_weight

        # Determine recommendation
        recommendation = self._get_recommendation(weighted_score, weighted_confidence)

        return {
            'ticker': ticker,
            'combined_score': round(weighted_score, 3),
            'confidence': round(weighted_confidence, 3),
            'signal_count': len(ticker_opps),
            'breakdown': breakdown,
            'recommendation': recommendation,
        }

    def _get_recommendation(self, score: float, confidence: float) -> str:
        """Determine recommendation based on combined score and confidence."""
        if score >= 0.7 and confidence >= 0.7:
            return 'STRONG BUY'
        elif score >= 0.6 and confidence >= 0.5:
            return 'BUY'
        elif score <= 0.3 and confidence >= 0.7:
            return 'STRONG SELL'
        elif score <= 0.4 and confidence >= 0.5:
            return 'SELL'
        else:
            return 'HOLD'

    def get_combined_opportunities_df(self) -> pd.DataFrame:
        """
        Get combined opportunities as a DataFrame with scores from all systems.

        Returns:
            DataFrame with combined opportunity data
        """
        opportunities = self.get_all_opportunities()

        if not opportunities:
            return pd.DataFrame()

        # Get unique tickers
        tickers = set(o.ticker for o in opportunities if o.ticker)

        # Calculate combined scores for each ticker
        combined_data = []
        for ticker in tickers:
            ticker_opps = [o for o in opportunities if o.ticker == ticker]
            combined = self.calculate_combined_score(ticker, opportunities)

            # Get company name
            company_name = next(
                (o.company_name for o in ticker_opps if o.company_name),
                ticker
            )

            combined_data.append({
                'ticker': ticker,
                'company_name': company_name,
                'combined_score': combined['combined_score'],
                'confidence': combined['confidence'],
                'signal_count': combined['signal_count'],
                'recommendation': combined['recommendation'],
                'clinical_score': combined['breakdown'].get('clinical_trials', {}).get('score', 0),
                'patent_score': combined['breakdown'].get('patent_intelligence', {}).get('score', 0),
                'insider_score': combined['breakdown'].get('insider_hiring', {}).get('score', 0),
            })

        df = pd.DataFrame(combined_data)

        # Sort by combined score descending
        if not df.empty:
            df = df.sort_values('combined_score', ascending=False)

        return df

    def get_top_opportunities(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N combined opportunities.

        Args:
            n: Number of opportunities to return

        Returns:
            DataFrame with top opportunities
        """
        df = self.get_combined_opportunities_df()
        return df.head(n) if not df.empty else df

    def get_alerts(
        self,
        score_threshold: float = 0.7,
        confidence_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Get high-confidence alert signals.

        Args:
            score_threshold: Minimum score for alerts
            confidence_threshold: Minimum confidence for alerts

        Returns:
            List of alert dictionaries
        """
        df = self.get_combined_opportunities_df()
        alerts = []

        if df.empty:
            return alerts

        # Filter for high-score opportunities
        high_score = df[
            (df['combined_score'] >= score_threshold) &
            (df['confidence'] >= confidence_threshold)
        ]

        for _, row in high_score.iterrows():
            alerts.append({
                'type': 'BULLISH',
                'ticker': row['ticker'],
                'company_name': row['company_name'],
                'combined_score': row['combined_score'],
                'confidence': row['confidence'],
                'recommendation': row['recommendation'],
                'message': f"High confidence bullish signal for {row['ticker']}",
            })

        # Filter for low-score opportunities (bearish)
        low_score = df[
            (df['combined_score'] <= (1 - score_threshold)) &
            (df['confidence'] >= confidence_threshold)
        ]

        for _, row in low_score.iterrows():
            alerts.append({
                'type': 'BEARISH',
                'ticker': row['ticker'],
                'company_name': row['company_name'],
                'combined_score': row['combined_score'],
                'confidence': row['confidence'],
                'recommendation': row['recommendation'],
                'message': f"High confidence bearish signal for {row['ticker']}",
            })

        return alerts
