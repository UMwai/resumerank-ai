"""
AI-Powered Glassdoor Analysis Enhancement

Deep sentiment analysis using Claude AI for Glassdoor reviews.
Extracts:
- Pipeline/R&D concerns and optimism
- Layoff/restructuring signals
- Management/leadership sentiment
- Commercial readiness indicators
- Culture and morale trends
- Confidence scores for each signal
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_config
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class DetailedSentimentAnalysis:
    """Comprehensive AI-powered sentiment analysis."""
    # Core sentiment
    overall_sentiment: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0

    # Pipeline/R&D signals
    pipeline_sentiment: float
    pipeline_concerns: List[str]
    pipeline_positives: List[str]
    clinical_trial_mentions: int
    fda_mentions: int

    # Layoff/restructuring signals
    layoff_risk_score: float  # 0.0 to 1.0
    layoff_mentions: int
    restructuring_mentions: int
    layoff_indicators: List[str]

    # Management signals
    management_sentiment: float
    leadership_concerns: List[str]
    leadership_strengths: List[str]
    ceo_mentions: int
    executive_turnover_mentions: int

    # Commercial signals
    commercial_readiness: float  # 0.0 to 1.0
    launch_indicators: List[str]
    commercial_team_sentiment: str

    # Culture signals
    culture_score: float
    morale_trend: str  # 'improving', 'declining', 'stable'
    burnout_indicators: List[str]
    work_life_balance: float

    # Overall signal
    investment_signal: str  # 'bullish', 'bearish', 'neutral'
    signal_strength: int  # -10 to +10
    key_themes: List[str]
    summary: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'overall_sentiment': self.overall_sentiment,
            'confidence': self.confidence,
            'pipeline': {
                'sentiment': self.pipeline_sentiment,
                'concerns': self.pipeline_concerns,
                'positives': self.pipeline_positives,
                'clinical_trial_mentions': self.clinical_trial_mentions,
                'fda_mentions': self.fda_mentions,
            },
            'layoff_risk': {
                'score': self.layoff_risk_score,
                'mentions': self.layoff_mentions,
                'restructuring_mentions': self.restructuring_mentions,
                'indicators': self.layoff_indicators,
            },
            'management': {
                'sentiment': self.management_sentiment,
                'concerns': self.leadership_concerns,
                'strengths': self.leadership_strengths,
                'ceo_mentions': self.ceo_mentions,
                'turnover_mentions': self.executive_turnover_mentions,
            },
            'commercial': {
                'readiness': self.commercial_readiness,
                'indicators': self.launch_indicators,
                'team_sentiment': self.commercial_team_sentiment,
            },
            'culture': {
                'score': self.culture_score,
                'morale_trend': self.morale_trend,
                'burnout_indicators': self.burnout_indicators,
                'work_life_balance': self.work_life_balance,
            },
            'investment_signal': self.investment_signal,
            'signal_strength': self.signal_strength,
            'key_themes': self.key_themes,
            'summary': self.summary,
        }


@dataclass
class SentimentTrend:
    """Sentiment trend over time."""
    ticker: str
    analysis_date: date
    current_sentiment: float
    previous_sentiment: float  # 30 days ago
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_magnitude: float  # % change
    trend_confidence: float
    notable_changes: List[str]


class GlassdoorAIAnalyzer:
    """
    AI-powered deep analysis of Glassdoor reviews.

    Uses Claude to extract nuanced signals from employee reviews
    that are particularly relevant to biotech investing.
    """

    ANALYSIS_PROMPT_TEMPLATE = """You are an expert analyst specializing in biotech company sentiment analysis for investment purposes.

Analyze these Glassdoor reviews for {company_name} ({ticker}) and extract investment-relevant signals.

REVIEWS:
{reviews_text}

Provide a detailed analysis in JSON format with these exact fields:

{{
    "overall_sentiment": <float -1.0 to 1.0, negative to positive>,
    "confidence": <float 0.0 to 1.0, how confident in analysis>,

    "pipeline_sentiment": <float -1.0 to 1.0, sentiment about drug pipeline/R&D>,
    "pipeline_concerns": [<list of specific concerns about pipeline, trials, science>],
    "pipeline_positives": [<list of positive signals about pipeline progress>],
    "clinical_trial_mentions": <int, count of clinical trial/phase references>,
    "fda_mentions": <int, count of FDA/regulatory references>,

    "layoff_risk_score": <float 0.0 to 1.0, likelihood of layoffs>,
    "layoff_mentions": <int, explicit layoff/RIF mentions>,
    "restructuring_mentions": <int, reorganization/restructuring mentions>,
    "layoff_indicators": [<list of indirect layoff signals like hiring freeze, cost cutting>],

    "management_sentiment": <float -1.0 to 1.0, sentiment about leadership>,
    "leadership_concerns": [<specific leadership/management problems mentioned>],
    "leadership_strengths": [<positive leadership qualities mentioned>],
    "ceo_mentions": <int, times CEO specifically mentioned>,
    "executive_turnover_mentions": <int, mentions of executives leaving>,

    "commercial_readiness": <float 0.0 to 1.0, how ready for commercial launch>,
    "launch_indicators": [<signals of preparing for product launch>],
    "commercial_team_sentiment": <"positive", "negative", or "neutral">,

    "culture_score": <float -1.0 to 1.0, company culture quality>,
    "morale_trend": <"improving", "declining", or "stable">,
    "burnout_indicators": [<signs of employee burnout or overwork>],
    "work_life_balance": <float -1.0 to 1.0>,

    "investment_signal": <"bullish", "bearish", or "neutral">,
    "signal_strength": <int -10 to +10, how strong the investment signal is>,
    "key_themes": [<3-5 main themes from reviews>],
    "summary": "<2-3 sentence investment-relevant summary>"
}}

Focus on:
1. Pipeline/clinical trial progress - critical for biotech
2. Signs of layoffs or cost cutting (very bearish)
3. Leadership stability and competence
4. Commercial preparation for drug launches (bullish)
5. Employee morale as leading indicator

Be specific and actionable. If insufficient data for a metric, use reasonable defaults."""

    TREND_ANALYSIS_PROMPT = """Analyze the sentiment trend for {company_name} ({ticker}) over time.

CURRENT PERIOD (Last 30 days):
{current_reviews}

PREVIOUS PERIOD (30-60 days ago):
{previous_reviews}

Compare the two periods and identify:
1. Direction of sentiment change
2. Magnitude of change
3. Key factors driving the change
4. Any emerging concerns or improvements

Provide analysis in JSON format:
{{
    "current_sentiment": <float -1.0 to 1.0>,
    "previous_sentiment": <float -1.0 to 1.0>,
    "trend_direction": <"improving", "declining", or "stable">,
    "trend_magnitude": <float percentage change>,
    "trend_confidence": <float 0.0 to 1.0>,
    "notable_changes": [<list of key changes between periods>],
    "summary": "<brief description of trend>"
}}"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize AI analyzer."""
        self.config = get_config(config_path) if config_path else None
        self._client = None

    @property
    def client(self):
        """Lazy initialize Anthropic client."""
        if self._client is None:
            api_key = None
            if self.config:
                api_key = self.config.anthropic_api_key
            if not api_key:
                api_key = os.environ.get('ANTHROPIC_API_KEY')

            if not api_key:
                logger.warning("No Anthropic API key found")
                return None

            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                logger.error("anthropic package not installed")
                return None

        return self._client

    def analyze_reviews(
        self,
        ticker: str,
        reviews: List[Dict],
        company_name: Optional[str] = None
    ) -> DetailedSentimentAnalysis:
        """
        Perform deep AI analysis on a batch of reviews.

        Args:
            ticker: Company ticker
            reviews: List of review dicts with 'pros', 'cons', 'rating', etc.
            company_name: Company name for context

        Returns:
            DetailedSentimentAnalysis object
        """
        if not reviews:
            return self._empty_analysis()

        if not self.client:
            return self._fallback_analysis(reviews)

        # Format reviews for prompt
        reviews_text = self._format_reviews(reviews)

        # Build prompt
        prompt = self.ANALYSIS_PROMPT_TEMPLATE.format(
            company_name=company_name or ticker,
            ticker=ticker,
            reviews_text=reviews_text
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            # Parse JSON from response
            analysis_data = self._parse_json_response(response_text)

            return DetailedSentimentAnalysis(
                overall_sentiment=float(analysis_data.get('overall_sentiment', 0)),
                confidence=float(analysis_data.get('confidence', 0.5)),
                pipeline_sentiment=float(analysis_data.get('pipeline_sentiment', 0)),
                pipeline_concerns=analysis_data.get('pipeline_concerns', []),
                pipeline_positives=analysis_data.get('pipeline_positives', []),
                clinical_trial_mentions=int(analysis_data.get('clinical_trial_mentions', 0)),
                fda_mentions=int(analysis_data.get('fda_mentions', 0)),
                layoff_risk_score=float(analysis_data.get('layoff_risk_score', 0)),
                layoff_mentions=int(analysis_data.get('layoff_mentions', 0)),
                restructuring_mentions=int(analysis_data.get('restructuring_mentions', 0)),
                layoff_indicators=analysis_data.get('layoff_indicators', []),
                management_sentiment=float(analysis_data.get('management_sentiment', 0)),
                leadership_concerns=analysis_data.get('leadership_concerns', []),
                leadership_strengths=analysis_data.get('leadership_strengths', []),
                ceo_mentions=int(analysis_data.get('ceo_mentions', 0)),
                executive_turnover_mentions=int(analysis_data.get('executive_turnover_mentions', 0)),
                commercial_readiness=float(analysis_data.get('commercial_readiness', 0)),
                launch_indicators=analysis_data.get('launch_indicators', []),
                commercial_team_sentiment=analysis_data.get('commercial_team_sentiment', 'neutral'),
                culture_score=float(analysis_data.get('culture_score', 0)),
                morale_trend=analysis_data.get('morale_trend', 'stable'),
                burnout_indicators=analysis_data.get('burnout_indicators', []),
                work_life_balance=float(analysis_data.get('work_life_balance', 0)),
                investment_signal=analysis_data.get('investment_signal', 'neutral'),
                signal_strength=int(analysis_data.get('signal_strength', 0)),
                key_themes=analysis_data.get('key_themes', []),
                summary=analysis_data.get('summary', 'Analysis not available')
            )

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._fallback_analysis(reviews)

    def analyze_trend(
        self,
        ticker: str,
        current_reviews: List[Dict],
        previous_reviews: List[Dict],
        company_name: Optional[str] = None
    ) -> SentimentTrend:
        """
        Analyze sentiment trend over time.

        Args:
            ticker: Company ticker
            current_reviews: Reviews from last 30 days
            previous_reviews: Reviews from 30-60 days ago
            company_name: Company name

        Returns:
            SentimentTrend object
        """
        if not current_reviews and not previous_reviews:
            return SentimentTrend(
                ticker=ticker,
                analysis_date=date.today(),
                current_sentiment=0,
                previous_sentiment=0,
                trend_direction='stable',
                trend_magnitude=0,
                trend_confidence=0,
                notable_changes=[]
            )

        if not self.client:
            return self._fallback_trend(ticker, current_reviews, previous_reviews)

        current_text = self._format_reviews(current_reviews[:10])
        previous_text = self._format_reviews(previous_reviews[:10])

        prompt = self.TREND_ANALYSIS_PROMPT.format(
            company_name=company_name or ticker,
            ticker=ticker,
            current_reviews=current_text if current_text else "No recent reviews",
            previous_reviews=previous_text if previous_text else "No previous reviews"
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            trend_data = self._parse_json_response(response_text)

            return SentimentTrend(
                ticker=ticker,
                analysis_date=date.today(),
                current_sentiment=float(trend_data.get('current_sentiment', 0)),
                previous_sentiment=float(trend_data.get('previous_sentiment', 0)),
                trend_direction=trend_data.get('trend_direction', 'stable'),
                trend_magnitude=float(trend_data.get('trend_magnitude', 0)),
                trend_confidence=float(trend_data.get('trend_confidence', 0.5)),
                notable_changes=trend_data.get('notable_changes', [])
            )

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return self._fallback_trend(ticker, current_reviews, previous_reviews)

    def _format_reviews(self, reviews: List[Dict], max_length: int = 4000) -> str:
        """Format reviews for the prompt."""
        formatted = []

        for i, review in enumerate(reviews, 1):
            text = f"Review {i}:\n"
            if review.get('rating'):
                text += f"Rating: {review['rating']}/5\n"
            if review.get('job_title'):
                text += f"Role: {review['job_title']}\n"
            if review.get('employment_status'):
                text += f"Status: {review['employment_status']}\n"
            if review.get('pros'):
                text += f"Pros: {review['pros'][:500]}\n"
            if review.get('cons'):
                text += f"Cons: {review['cons'][:500]}\n"
            if review.get('review_text') and not review.get('pros'):
                text += f"Review: {review['review_text'][:500]}\n"

            formatted.append(text)

        result = "\n---\n".join(formatted)

        # Truncate if too long
        if len(result) > max_length:
            result = result[:max_length] + "\n[Truncated for length]"

        return result

    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON from Claude response."""
        # Try to extract JSON block
        json_match = re.search(r'\{[\s\S]*\}', response_text)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try parsing the whole response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from response")
            return {}

    def _empty_analysis(self) -> DetailedSentimentAnalysis:
        """Return empty analysis."""
        return DetailedSentimentAnalysis(
            overall_sentiment=0,
            confidence=0,
            pipeline_sentiment=0,
            pipeline_concerns=[],
            pipeline_positives=[],
            clinical_trial_mentions=0,
            fda_mentions=0,
            layoff_risk_score=0,
            layoff_mentions=0,
            restructuring_mentions=0,
            layoff_indicators=[],
            management_sentiment=0,
            leadership_concerns=[],
            leadership_strengths=[],
            ceo_mentions=0,
            executive_turnover_mentions=0,
            commercial_readiness=0,
            launch_indicators=[],
            commercial_team_sentiment='neutral',
            culture_score=0,
            morale_trend='stable',
            burnout_indicators=[],
            work_life_balance=0,
            investment_signal='neutral',
            signal_strength=0,
            key_themes=[],
            summary='No reviews available for analysis'
        )

    def _fallback_analysis(self, reviews: List[Dict]) -> DetailedSentimentAnalysis:
        """Keyword-based fallback when AI not available."""
        if not reviews:
            return self._empty_analysis()

        # Combine all text
        all_text = ""
        ratings = []
        for r in reviews:
            all_text += f" {r.get('pros', '')} {r.get('cons', '')} {r.get('review_text', '')}"
            if r.get('rating'):
                ratings.append(float(r['rating']))

        all_text_lower = all_text.lower()

        # Calculate basic sentiment from ratings
        avg_rating = sum(ratings) / len(ratings) if ratings else 3
        overall_sentiment = (avg_rating - 3) / 2  # Convert to -1 to 1

        # Keyword detection
        layoff_keywords = ['layoff', 'laid off', 'rif', 'restructur', 'downsiz', 'cut']
        pipeline_keywords = ['pipeline', 'clinical', 'phase', 'fda', 'trial', 'drug']
        management_keywords = ['leadership', 'management', 'ceo', 'executive', 'direction']
        commercial_keywords = ['commercial', 'launch', 'sales', 'marketing', 'msl']

        layoff_mentions = sum(1 for k in layoff_keywords if k in all_text_lower)
        pipeline_mentions = sum(1 for k in pipeline_keywords if k in all_text_lower)
        management_mentions = sum(1 for k in management_keywords if k in all_text_lower)
        commercial_mentions = sum(1 for k in commercial_keywords if k in all_text_lower)

        # Calculate signal
        signal_strength = int(overall_sentiment * 5)
        if layoff_mentions >= 3:
            signal_strength -= 4

        investment_signal = 'neutral'
        if signal_strength >= 3:
            investment_signal = 'bullish'
        elif signal_strength <= -3:
            investment_signal = 'bearish'

        return DetailedSentimentAnalysis(
            overall_sentiment=round(overall_sentiment, 2),
            confidence=0.5,  # Lower confidence for fallback
            pipeline_sentiment=0,
            pipeline_concerns=[],
            pipeline_positives=[],
            clinical_trial_mentions=pipeline_mentions,
            fda_mentions=0,
            layoff_risk_score=min(1.0, layoff_mentions * 0.25),
            layoff_mentions=layoff_mentions,
            restructuring_mentions=0,
            layoff_indicators=[],
            management_sentiment=0,
            leadership_concerns=[],
            leadership_strengths=[],
            ceo_mentions=0,
            executive_turnover_mentions=0,
            commercial_readiness=min(1.0, commercial_mentions * 0.2),
            launch_indicators=[],
            commercial_team_sentiment='neutral',
            culture_score=overall_sentiment * 0.8,
            morale_trend='stable',
            burnout_indicators=[],
            work_life_balance=0,
            investment_signal=investment_signal,
            signal_strength=signal_strength,
            key_themes=[],
            summary='Basic keyword analysis (AI not available)'
        )

    def _fallback_trend(
        self,
        ticker: str,
        current_reviews: List[Dict],
        previous_reviews: List[Dict]
    ) -> SentimentTrend:
        """Fallback trend analysis."""
        current_ratings = [float(r.get('rating', 3)) for r in current_reviews if r.get('rating')]
        previous_ratings = [float(r.get('rating', 3)) for r in previous_reviews if r.get('rating')]

        current_avg = sum(current_ratings) / len(current_ratings) if current_ratings else 3
        previous_avg = sum(previous_ratings) / len(previous_ratings) if previous_ratings else 3

        current_sentiment = (current_avg - 3) / 2
        previous_sentiment = (previous_avg - 3) / 2

        change = current_sentiment - previous_sentiment

        if change > 0.1:
            direction = 'improving'
        elif change < -0.1:
            direction = 'declining'
        else:
            direction = 'stable'

        return SentimentTrend(
            ticker=ticker,
            analysis_date=date.today(),
            current_sentiment=round(current_sentiment, 2),
            previous_sentiment=round(previous_sentiment, 2),
            trend_direction=direction,
            trend_magnitude=round(abs(change) * 100, 1),
            trend_confidence=0.4,
            notable_changes=[]
        )

    def generate_investment_report(
        self,
        analysis: DetailedSentimentAnalysis,
        ticker: str,
        company_name: Optional[str] = None
    ) -> str:
        """Generate a formatted investment report from analysis."""
        report = []
        report.append("=" * 60)
        report.append(f"GLASSDOOR SENTIMENT ANALYSIS: {company_name or ticker} ({ticker})")
        report.append("=" * 60)
        report.append("")

        # Overall signal
        signal_emoji = {
            'bullish': '[+]',
            'bearish': '[-]',
            'neutral': '[=]'
        }.get(analysis.investment_signal, '[=]')

        report.append(f"INVESTMENT SIGNAL: {signal_emoji} {analysis.investment_signal.upper()}")
        report.append(f"Signal Strength: {analysis.signal_strength:+d}/10")
        report.append(f"Confidence: {analysis.confidence:.0%}")
        report.append("")

        # Summary
        report.append("-" * 40)
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(analysis.summary)
        report.append("")

        # Pipeline
        report.append("-" * 40)
        report.append("PIPELINE / R&D")
        report.append("-" * 40)
        report.append(f"Pipeline Sentiment: {analysis.pipeline_sentiment:+.2f}")
        if analysis.pipeline_positives:
            report.append("Positives:")
            for p in analysis.pipeline_positives[:3]:
                report.append(f"  + {p}")
        if analysis.pipeline_concerns:
            report.append("Concerns:")
            for c in analysis.pipeline_concerns[:3]:
                report.append(f"  - {c}")
        report.append("")

        # Layoff Risk
        report.append("-" * 40)
        report.append("LAYOFF / RESTRUCTURING RISK")
        report.append("-" * 40)
        risk_level = "LOW" if analysis.layoff_risk_score < 0.3 else "MEDIUM" if analysis.layoff_risk_score < 0.6 else "HIGH"
        report.append(f"Risk Score: {analysis.layoff_risk_score:.0%} ({risk_level})")
        report.append(f"Direct Mentions: {analysis.layoff_mentions}")
        if analysis.layoff_indicators:
            report.append("Indicators:")
            for i in analysis.layoff_indicators[:3]:
                report.append(f"  ! {i}")
        report.append("")

        # Management
        report.append("-" * 40)
        report.append("LEADERSHIP / MANAGEMENT")
        report.append("-" * 40)
        report.append(f"Management Sentiment: {analysis.management_sentiment:+.2f}")
        if analysis.leadership_strengths:
            report.append("Strengths:")
            for s in analysis.leadership_strengths[:2]:
                report.append(f"  + {s}")
        if analysis.leadership_concerns:
            report.append("Concerns:")
            for c in analysis.leadership_concerns[:2]:
                report.append(f"  - {c}")
        report.append("")

        # Commercial Readiness
        report.append("-" * 40)
        report.append("COMMERCIAL READINESS")
        report.append("-" * 40)
        report.append(f"Readiness Score: {analysis.commercial_readiness:.0%}")
        if analysis.launch_indicators:
            report.append("Launch Indicators:")
            for i in analysis.launch_indicators[:3]:
                report.append(f"  * {i}")
        report.append("")

        # Culture
        report.append("-" * 40)
        report.append("CULTURE / MORALE")
        report.append("-" * 40)
        report.append(f"Culture Score: {analysis.culture_score:+.2f}")
        report.append(f"Morale Trend: {analysis.morale_trend.upper()}")
        report.append(f"Work-Life Balance: {analysis.work_life_balance:+.2f}")
        report.append("")

        # Key Themes
        if analysis.key_themes:
            report.append("-" * 40)
            report.append("KEY THEMES")
            report.append("-" * 40)
            for theme in analysis.key_themes:
                report.append(f"  - {theme}")
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)


if __name__ == '__main__':
    # Test the analyzer
    analyzer = GlassdoorAIAnalyzer()

    # Sample reviews
    test_reviews = [
        {
            'rating': 4,
            'job_title': 'Senior Scientist',
            'employment_status': 'Current',
            'pros': 'Great pipeline progress, exciting science, Phase 3 data coming soon.',
            'cons': 'Long hours, some management communication issues.',
        },
        {
            'rating': 3,
            'job_title': 'Clinical Research Associate',
            'employment_status': 'Current',
            'pros': 'Good benefits, talented colleagues.',
            'cons': 'Heard rumors of restructuring, uncertain future.',
        },
        {
            'rating': 5,
            'job_title': 'Commercial Lead',
            'employment_status': 'Current',
            'pros': 'Building commercial team for launch, exciting times.',
            'cons': 'Very fast paced.',
        },
    ]

    print("Testing Glassdoor AI Analyzer")
    print("=" * 50)

    analysis = analyzer.analyze_reviews('MRNA', test_reviews, 'Moderna')

    print(analyzer.generate_investment_report(analysis, 'MRNA', 'Moderna'))
