"""
Report Generator for Insider Activity + Hiring Signals System

Generates comprehensive signal reports in multiple formats:
- Console/Terminal output (using rich)
- HTML reports
- JSON export
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from tabulate import tabulate

from models.signal_scorer import SignalScore, SignalScorer
from utils.config import get_config
from utils.database import get_database
from utils.logger import setup_logger

logger = setup_logger(__name__)
console = Console()


class ReportGenerator:
    """
    Generates signal reports in various formats.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.db = get_database(config_path)
        self.scorer = SignalScorer(config_path)
        self.reports_dir = Path(__file__).parent.parent / 'data' / 'reports'
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_company_report(self, ticker: str) -> Dict[str, Any]:
        """
        Generate a detailed report for a single company.

        Returns:
            Dictionary with all signal data for the company
        """
        score = self.scorer.calculate_score(ticker)

        # Get recent transactions
        transactions = self.db.execute("""
            SELECT * FROM insider_transactions
            WHERE company_ticker = %s
            ORDER BY transaction_date DESC
            LIMIT 10
        """, (ticker,))

        # Get institutional holdings
        holdings = self.db.execute("""
            SELECT * FROM institutional_holdings
            WHERE company_ticker = %s
            ORDER BY quarter_end DESC
            LIMIT 20
        """, (ticker,))

        # Get job postings
        jobs = self.db.execute("""
            SELECT * FROM job_postings
            WHERE company_ticker = %s
            AND removal_date IS NULL
            ORDER BY first_seen_date DESC
            LIMIT 20
        """, (ticker,))

        return {
            'ticker': ticker,
            'score': score,
            'transactions': transactions,
            'holdings': holdings,
            'jobs': jobs,
            'generated_at': datetime.now().isoformat()
        }

    def generate_summary_report(self, tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a summary report for multiple companies.

        Args:
            tickers: List of tickers (defaults to watchlist)

        Returns:
            Dictionary with summary data
        """
        if tickers is None:
            tickers = self.config.watchlist

        scores = []
        for ticker in tickers:
            try:
                score = self.scorer.calculate_score(ticker)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Could not score {ticker}: {e}")

        # Sort by composite score
        scores.sort(key=lambda s: s.composite_score, reverse=True)

        # Categorize
        strong_buy = [s for s in scores if s.recommendation == 'STRONG BUY']
        buy = [s for s in scores if s.recommendation == 'BUY']
        neutral = [s for s in scores if s.recommendation == 'NEUTRAL']
        sell = [s for s in scores if s.recommendation == 'SELL']
        strong_sell = [s for s in scores if s.recommendation == 'STRONG SELL']

        return {
            'date': date.today().isoformat(),
            'total_companies': len(scores),
            'scores': scores,
            'by_recommendation': {
                'STRONG BUY': strong_buy,
                'BUY': buy,
                'NEUTRAL': neutral,
                'SELL': sell,
                'STRONG SELL': strong_sell
            },
            'top_5_bullish': scores[:5],
            'top_5_bearish': scores[-5:][::-1] if len(scores) >= 5 else [],
            'generated_at': datetime.now().isoformat()
        }

    def print_console_report(self, tickers: Optional[List[str]] = None):
        """
        Print a formatted report to the console using rich.

        Args:
            tickers: List of tickers (defaults to watchlist)
        """
        report = self.generate_summary_report(tickers)

        # Header
        console.print()
        console.print(Panel.fit(
            "[bold blue]Biotech Insider + Hiring Signals Report[/bold blue]\n"
            f"[dim]{report['date']} | {report['total_companies']} companies analyzed[/dim]",
            border_style="blue"
        ))

        # Summary stats
        summary_table = Table(title="Summary", show_header=False)
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("Count", justify="right")

        summary_table.add_row("STRONG BUY", str(len(report['by_recommendation']['STRONG BUY'])))
        summary_table.add_row("BUY", str(len(report['by_recommendation']['BUY'])))
        summary_table.add_row("NEUTRAL", str(len(report['by_recommendation']['NEUTRAL'])))
        summary_table.add_row("SELL", str(len(report['by_recommendation']['SELL'])))
        summary_table.add_row("STRONG SELL", str(len(report['by_recommendation']['STRONG SELL'])))

        console.print(summary_table)
        console.print()

        # Top Bullish
        if report['top_5_bullish']:
            console.print("[bold green]Top Bullish Signals[/bold green]")
            bullish_table = Table()
            bullish_table.add_column("Ticker", style="cyan bold")
            bullish_table.add_column("Score", justify="right")
            bullish_table.add_column("Confidence", justify="right")
            bullish_table.add_column("Signals", justify="right")
            bullish_table.add_column("Recommendation", style="green")
            bullish_table.add_column("Top Signal")

            for score in report['top_5_bullish']:
                if score.composite_score > 0:
                    top_signal = score.contributing_signals[0].description if score.contributing_signals else '-'
                    bullish_table.add_row(
                        score.company_ticker,
                        f"+{score.composite_score:.1f}",
                        f"{score.confidence*100:.0f}%",
                        str(score.signal_count),
                        score.recommendation,
                        top_signal[:50] + "..." if len(top_signal) > 50 else top_signal
                    )

            console.print(bullish_table)
            console.print()

        # Top Bearish
        bearish = [s for s in report['scores'] if s.composite_score < -1]
        if bearish:
            console.print("[bold red]Top Bearish Signals[/bold red]")
            bearish_table = Table()
            bearish_table.add_column("Ticker", style="cyan bold")
            bearish_table.add_column("Score", justify="right", style="red")
            bearish_table.add_column("Confidence", justify="right")
            bearish_table.add_column("Signals", justify="right")
            bearish_table.add_column("Recommendation", style="red")
            bearish_table.add_column("Top Signal")

            for score in bearish[-5:]:
                top_signal = score.contributing_signals[0].description if score.contributing_signals else '-'
                bearish_table.add_row(
                    score.company_ticker,
                    f"{score.composite_score:.1f}",
                    f"{score.confidence*100:.0f}%",
                    str(score.signal_count),
                    score.recommendation,
                    top_signal[:50] + "..." if len(top_signal) > 50 else top_signal
                )

            console.print(bearish_table)
            console.print()

        # Detailed breakdown for top 3
        console.print("[bold]Signal Breakdown (Top 3)[/bold]")
        for i, score in enumerate(report['top_5_bullish'][:3], 1):
            self._print_company_detail(score)

    def _print_company_detail(self, score: SignalScore):
        """Print detailed signal breakdown for a company."""
        # Header
        rec_color = {
            'STRONG BUY': 'green',
            'BUY': 'green',
            'NEUTRAL': 'yellow',
            'SELL': 'red',
            'STRONG SELL': 'red'
        }.get(score.recommendation, 'white')

        console.print(Panel(
            f"[bold]{score.company_ticker}[/bold] | "
            f"Score: [bold]{score.composite_score:+.2f}[/bold] | "
            f"[{rec_color}]{score.recommendation}[/{rec_color}]",
            expand=False
        ))

        # Score breakdown
        breakdown_table = Table(show_header=False, box=None)
        breakdown_table.add_column("Category", width=20)
        breakdown_table.add_column("Score", justify="right", width=10)

        breakdown_table.add_row("Insider Score", f"{score.insider_score:+.2f}")
        breakdown_table.add_row("Institutional Score", f"{score.institutional_score:+.2f}")
        breakdown_table.add_row("Hiring Score", f"{score.hiring_score:+.2f}")

        console.print(breakdown_table)

        # Contributing signals
        if score.contributing_signals:
            console.print("\n[dim]Contributing Signals:[/dim]")
            for signal in score.contributing_signals[:5]:
                weight_color = "green" if signal.weighted_score > 0 else "red"
                console.print(
                    f"  [{weight_color}]{signal.weighted_score:+.2f}[/{weight_color}] "
                    f"[dim]({signal.category})[/dim] {signal.description}"
                )

        console.print()

    def generate_example_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate an example signal report showing composite scores.

        This is useful for demonstrating the system's output format.

        Args:
            output_file: Optional path to save the report

        Returns:
            The report as a string
        """
        # Sample data for demonstration (would normally come from scraped data)
        sample_companies = [
            {
                'ticker': 'MRNA',
                'name': 'Moderna Inc',
                'score': 6.5,
                'confidence': 0.75,
                'recommendation': 'STRONG BUY',
                'insider_score': 3.5,
                'institutional_score': 2.0,
                'hiring_score': 3.0,
                'signals': [
                    {'type': 'CEO_PURCHASE', 'desc': 'CEO bought $200K of shares', 'weight': 3.75},
                    {'type': 'COMMERCIAL_BUILDUP', 'desc': '8 commercial roles posted', 'weight': 3.35},
                    {'type': 'FUND_NEW_POSITION', 'desc': 'Baker Bros initiated position', 'weight': 2.50},
                ]
            },
            {
                'ticker': 'VRTX',
                'name': 'Vertex Pharmaceuticals',
                'score': 4.2,
                'confidence': 0.65,
                'recommendation': 'BUY',
                'insider_score': 2.5,
                'institutional_score': 1.5,
                'hiring_score': 2.0,
                'signals': [
                    {'type': 'CFO_PURCHASE', 'desc': 'CFO bought $100K of shares', 'weight': 2.50},
                    {'type': 'CLINICAL_EXPANSION', 'desc': '5 clinical operations roles', 'weight': 2.00},
                    {'type': 'FUND_INCREASE_50', 'desc': 'RA Capital increased 60%', 'weight': 1.90},
                ]
            },
            {
                'ticker': 'CRSP',
                'name': 'CRISPR Therapeutics',
                'score': 3.8,
                'confidence': 0.60,
                'recommendation': 'BUY',
                'insider_score': 1.5,
                'institutional_score': 2.0,
                'hiring_score': 1.5,
                'signals': [
                    {'type': 'FUND_NEW_POSITION', 'desc': 'Perceptive initiated position', 'weight': 2.00},
                    {'type': 'DIRECTOR_PURCHASE', 'desc': 'Director bought $75K', 'weight': 1.50},
                    {'type': 'REGULATORY_EXPANSION', 'desc': 'Regulatory team hiring', 'weight': 1.20},
                ]
            },
            {
                'ticker': 'BEAM',
                'name': 'Beam Therapeutics',
                'score': 1.2,
                'confidence': 0.45,
                'recommendation': 'NEUTRAL',
                'insider_score': 0.5,
                'institutional_score': 0.5,
                'hiring_score': 0.5,
                'signals': [
                    {'type': 'OFFICER_PURCHASE', 'desc': 'Officer bought $50K', 'weight': 0.80},
                    {'type': 'FUND_INCREASE_25', 'desc': 'Tang Capital increased 30%', 'weight': 0.70},
                ]
            },
            {
                'ticker': 'EDIT',
                'name': 'Editas Medicine',
                'score': -2.5,
                'confidence': 0.55,
                'recommendation': 'NEUTRAL',
                'insider_score': -1.5,
                'institutional_score': -1.0,
                'hiring_score': 0.0,
                'signals': [
                    {'type': 'CFO_SALE', 'desc': 'CFO sold $150K of shares', 'weight': -2.00},
                    {'type': 'FUND_DECREASE_25', 'desc': 'OrbiMed decreased 30%', 'weight': -1.00},
                ]
            },
            {
                'ticker': 'BLUE',
                'name': 'Bluebird Bio',
                'score': -5.2,
                'confidence': 0.70,
                'recommendation': 'SELL',
                'insider_score': -2.5,
                'institutional_score': -1.5,
                'hiring_score': -2.0,
                'signals': [
                    {'type': 'MULTIPLE_INSIDER_SELL', 'desc': '3 insiders sold in 30 days', 'weight': -3.00},
                    {'type': 'FUND_EXIT', 'desc': 'Boxer Capital exited position', 'weight': -2.00},
                    {'type': 'HIRING_FREEZE', 'desc': '8 job postings removed', 'weight': -1.50},
                ]
            },
            {
                'ticker': 'SAGE',
                'name': 'Sage Therapeutics',
                'score': -6.8,
                'confidence': 0.75,
                'recommendation': 'STRONG SELL',
                'insider_score': -3.0,
                'institutional_score': -2.5,
                'hiring_score': -2.5,
                'signals': [
                    {'type': 'CEO_SALE', 'desc': 'CEO sold 15% of holdings', 'weight': -3.00},
                    {'type': 'MULTIPLE_FUNDS_EXIT', 'desc': '3 funds exited positions', 'weight': -3.00},
                    {'type': 'HIRING_FREEZE', 'desc': '12 job postings removed', 'weight': -2.00},
                ]
            },
        ]

        # Generate report text
        lines = []
        lines.append("=" * 80)
        lines.append("BIOTECH INSIDER + HIRING SIGNALS REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")

        lines.append("SCORING METHODOLOGY")
        lines.append("-" * 40)
        lines.append("Composite Score: Weighted sum of all signals with time-decay")
        lines.append("  - Recent signals weighted more heavily (30-day half-life)")
        lines.append("  - Score range: -10 (very bearish) to +10 (very bullish)")
        lines.append("")
        lines.append("Confidence: Based on signal count and quality")
        lines.append("  - Range: 0.0 (low) to 1.0 (high)")
        lines.append("")
        lines.append("Recommendations:")
        lines.append("  STRONG BUY:  Score >= 6.0, Confidence >= 0.70")
        lines.append("  BUY:         Score >= 3.0, Confidence >= 0.50")
        lines.append("  NEUTRAL:     Between BUY and SELL thresholds")
        lines.append("  SELL:        Score <= -3.0, Confidence >= 0.50")
        lines.append("  STRONG SELL: Score <= -6.0, Confidence >= 0.70")
        lines.append("")

        lines.append("=" * 80)
        lines.append("SIGNAL SUMMARY")
        lines.append("=" * 80)
        lines.append("")

        # Summary table
        headers = ["Ticker", "Score", "Conf", "Rec", "Insider", "Inst", "Hiring"]
        rows = []
        for c in sample_companies:
            rows.append([
                c['ticker'],
                f"{c['score']:+.1f}",
                f"{c['confidence']:.0%}",
                c['recommendation'],
                f"{c['insider_score']:+.1f}",
                f"{c['institutional_score']:+.1f}",
                f"{c['hiring_score']:+.1f}"
            ])

        lines.append(tabulate(rows, headers=headers, tablefmt="simple"))
        lines.append("")

        lines.append("=" * 80)
        lines.append("DETAILED SIGNAL BREAKDOWN")
        lines.append("=" * 80)

        for c in sample_companies:
            lines.append("")
            lines.append(f"--- {c['ticker']} ({c['name']}) ---")
            lines.append(f"Composite Score: {c['score']:+.2f}")
            lines.append(f"Confidence: {c['confidence']:.0%}")
            lines.append(f"Recommendation: {c['recommendation']}")
            lines.append("")
            lines.append("Score Breakdown:")
            lines.append(f"  Insider Score:       {c['insider_score']:+.2f}")
            lines.append(f"  Institutional Score: {c['institutional_score']:+.2f}")
            lines.append(f"  Hiring Score:        {c['hiring_score']:+.2f}")
            lines.append("")
            lines.append("Contributing Signals:")
            for sig in c['signals']:
                sign = "+" if sig['weight'] > 0 else ""
                lines.append(f"  [{sign}{sig['weight']:.2f}] {sig['type']}: {sig['desc']}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("SIGNAL WEIGHTS REFERENCE")
        lines.append("=" * 80)
        lines.append("")
        lines.append("BULLISH SIGNALS:")
        lines.append("  +5: CEO buys >$100K, CMO buys (clinical confidence)")
        lines.append("  +6: Multiple insiders buy in 30 days")
        lines.append("  +5: Top fund initiates new position")
        lines.append("  +5: 5+ commercial roles posted (launch prep)")
        lines.append("  +4: VP Manufacturing hired, Clinical expansion")
        lines.append("")
        lines.append("BEARISH SIGNALS:")
        lines.append("  -5: CFO sells before quarter end")
        lines.append("  -6: Multiple insiders sell in 30 days")
        lines.append("  -5: Top fund exits position completely")
        lines.append("  -4: Hiring freeze (job postings removed)")
        lines.append("  -6: CMO departure")
        lines.append("")
        lines.append("=" * 80)
        lines.append("DISCLAIMER")
        lines.append("=" * 80)
        lines.append("")
        lines.append("This report is for informational purposes only and does not")
        lines.append("constitute investment advice. The signals and scores are based")
        lines.append("on publicly available information and historical patterns.")
        lines.append("Past performance does not guarantee future results.")
        lines.append("")
        lines.append("Always conduct your own due diligence before making investment")
        lines.append("decisions. The information may be incomplete or inaccurate.")
        lines.append("")

        report_text = "\n".join(lines)

        # Save if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")

        return report_text

    def export_json(self, tickers: Optional[List[str]] = None,
                    output_file: Optional[str] = None) -> Dict:
        """
        Export signal data as JSON.

        Args:
            tickers: List of tickers
            output_file: Optional path to save JSON

        Returns:
            Dictionary with all signal data
        """
        report = self.generate_summary_report(tickers)

        # Convert SignalScore objects to dictionaries
        def score_to_dict(score: SignalScore) -> Dict:
            return {
                'ticker': score.company_ticker,
                'score_date': str(score.score_date),
                'composite_score': score.composite_score,
                'confidence': score.confidence,
                'signal_count': score.signal_count,
                'insider_score': score.insider_score,
                'institutional_score': score.institutional_score,
                'hiring_score': score.hiring_score,
                'sentiment_score': score.sentiment_score,
                'recommendation': score.recommendation,
                'signals': [
                    {
                        'date': str(s.signal_date),
                        'category': s.category,
                        'type': s.signal_type,
                        'description': s.description,
                        'raw_weight': s.raw_weight,
                        'weighted_score': s.weighted_score
                    }
                    for s in score.contributing_signals
                ]
            }

        export_data = {
            'generated_at': report['generated_at'],
            'date': report['date'],
            'total_companies': report['total_companies'],
            'summary': {
                'strong_buy': len(report['by_recommendation']['STRONG BUY']),
                'buy': len(report['by_recommendation']['BUY']),
                'neutral': len(report['by_recommendation']['NEUTRAL']),
                'sell': len(report['by_recommendation']['SELL']),
                'strong_sell': len(report['by_recommendation']['STRONG SELL']),
            },
            'scores': [score_to_dict(s) for s in report['scores']]
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"JSON exported to {output_file}")

        return export_data


if __name__ == '__main__':
    # Generate example report
    generator = ReportGenerator()

    # Print to console
    print("\n" + "=" * 80)
    print("EXAMPLE SIGNAL REPORT")
    print("=" * 80)

    report = generator.generate_example_report(
        output_file='data/reports/example_signal_report.txt'
    )
    print(report)
