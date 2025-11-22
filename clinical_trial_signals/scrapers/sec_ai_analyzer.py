"""
AI-Powered SEC Filing Analysis for Clinical Trial Signal Detection System.

Uses Anthropic Claude API to analyze 8-K filings for:
- Sentiment analysis (bullish/bearish) from management language
- Hedging language vs confident language detection
- Numeric sentiment scoring from -10 to +10
- Clinical trial specific insights

This module provides high-ROI signal generation by extracting nuanced
sentiment and forward-looking statements from SEC filings.
"""
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config

logger = logging.getLogger(__name__)


class SentimentCategory(Enum):
    """Sentiment classification categories."""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class LanguagePattern:
    """Detected language pattern in filing."""
    pattern_type: str  # "hedging", "confident", "forward_looking", "risk_warning"
    text_excerpt: str
    confidence: float  # 0.0 to 1.0


@dataclass
class AIAnalysisResult:
    """Result of AI-powered SEC filing analysis."""
    filing_id: str
    sentiment_score: float  # -10 to +10
    sentiment_category: SentimentCategory
    confidence: float  # 0.0 to 1.0
    hedging_score: float  # 0 to 1, higher = more hedging language
    confidence_language_score: float  # 0 to 1, higher = more confident language
    key_insights: List[str]
    trial_mentions: List[Dict[str, str]]
    forward_looking_statements: List[str]
    risk_factors: List[str]
    language_patterns: List[LanguagePattern]
    raw_analysis: Dict[str, Any]
    processing_time_ms: int
    model_used: str
    tokens_used: int


class SECAIAnalyzer:
    """
    AI-powered analyzer for SEC filings using Anthropic Claude API.

    Provides deep analysis of 8-K filings to extract:
    - Overall sentiment (bullish/bearish)
    - Management confidence levels
    - Hedging language indicators
    - Clinical trial specific insights
    - Forward-looking statement analysis
    """

    # Default model to use
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    # Rate limiting settings
    MAX_REQUESTS_PER_MINUTE = 50
    REQUEST_INTERVAL = 60.0 / MAX_REQUESTS_PER_MINUTE

    # Token limits
    MAX_INPUT_TOKENS = 100000
    MAX_OUTPUT_TOKENS = 4096

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the SEC AI Analyzer.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (defaults to claude-sonnet-4-20250514)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model or self.DEFAULT_MODEL
        self._last_request_time = 0
        self._client = None

        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set - AI analysis will be unavailable")

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    def _rate_limit_wait(self) -> None:
        """Enforce rate limiting for API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_INTERVAL:
            time.sleep(self.REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _build_analysis_prompt(self, filing_content: str, filing_metadata: Dict) -> str:
        """
        Build the prompt for SEC filing analysis.

        Args:
            filing_content: Raw text content of the filing
            filing_metadata: Metadata about the filing (company, date, type, etc.)

        Returns:
            Formatted prompt string
        """
        company = filing_metadata.get("company_name", "Unknown Company")
        ticker = filing_metadata.get("ticker", "N/A")
        filing_type = filing_metadata.get("filing_type", "8-K")
        filing_date = filing_metadata.get("filing_date", "Unknown")

        prompt = f"""Analyze this SEC {filing_type} filing from {company} ({ticker}) dated {filing_date}.

Focus specifically on clinical trial and drug development related content. Provide a comprehensive analysis in JSON format.

SEC FILING CONTENT:
---
{filing_content[:80000]}
---

Analyze the filing and respond with ONLY a valid JSON object (no markdown, no explanation) with this exact structure:

{{
    "sentiment_score": <float from -10 to +10, where -10 is extremely bearish, 0 is neutral, +10 is extremely bullish>,
    "sentiment_category": "<one of: very_bullish, bullish, neutral, bearish, very_bearish>",
    "confidence": <float from 0.0 to 1.0 indicating how confident you are in this analysis>,
    "hedging_score": <float from 0.0 to 1.0, higher means more hedging/cautionary language>,
    "confidence_language_score": <float from 0.0 to 1.0, higher means more confident/assertive language>,
    "key_insights": [
        "<insight 1 - most important finding>",
        "<insight 2>",
        "<insight 3 - up to 5 insights>"
    ],
    "trial_mentions": [
        {{
            "trial_name": "<name or NCT number if mentioned>",
            "phase": "<trial phase if mentioned>",
            "indication": "<disease/condition>",
            "status_update": "<what was said about the trial>",
            "sentiment": "<positive/negative/neutral>"
        }}
    ],
    "forward_looking_statements": [
        "<statement 1 about future expectations>",
        "<statement 2>"
    ],
    "risk_factors": [
        "<risk 1 mentioned>",
        "<risk 2>"
    ],
    "language_patterns": [
        {{
            "pattern_type": "<hedging|confident|forward_looking|risk_warning>",
            "text_excerpt": "<brief quote from filing>",
            "confidence": <float 0.0 to 1.0>
        }}
    ],
    "summary": "<2-3 sentence summary of the filing's implications for investors>"
}}

IMPORTANT ANALYSIS GUIDELINES:
1. Sentiment Score Scale:
   - +8 to +10: Major positive catalyst (e.g., met primary endpoint, FDA approval)
   - +4 to +7: Positive news (e.g., good interim data, enrollment ahead of schedule)
   - +1 to +3: Mildly positive (e.g., trial progressing normally)
   - -1 to +1: Neutral or unclear
   - -3 to -1: Mildly negative (e.g., minor delays, increased costs)
   - -7 to -4: Negative news (e.g., missed secondary endpoint, safety concerns)
   - -10 to -8: Major negative catalyst (e.g., failed primary endpoint, clinical hold)

2. Hedging Language Indicators:
   - "may", "might", "could potentially", "subject to"
   - "no assurance", "cannot guarantee", "uncertain"
   - Multiple risk disclaimers, extensive caveats

3. Confident Language Indicators:
   - "will", "expect", "confident", "on track"
   - Specific timelines, concrete milestones
   - Definitive statements about progress

4. Focus on:
   - Clinical trial updates and results
   - Regulatory interactions (FDA, EMA)
   - Management tone and confidence
   - Forward-looking statements
   - Material agreements and partnerships
"""
        return prompt

    def analyze_filing(
        self,
        filing_content: str,
        filing_metadata: Dict,
        filing_id: Optional[str] = None
    ) -> Optional[AIAnalysisResult]:
        """
        Analyze a SEC filing using Claude AI.

        Args:
            filing_content: Raw text content of the filing
            filing_metadata: Metadata about the filing
            filing_id: Optional identifier for the filing

        Returns:
            AIAnalysisResult with analysis results, or None if analysis fails
        """
        if not self.api_key:
            logger.error("Cannot analyze filing - ANTHROPIC_API_KEY not configured")
            return None

        if not filing_content or len(filing_content.strip()) < 100:
            logger.warning("Filing content too short for meaningful analysis")
            return None

        filing_id = filing_id or filing_metadata.get("accession_number", "unknown")
        start_time = time.time()

        try:
            # Rate limit
            self._rate_limit_wait()

            # Build prompt
            prompt = self._build_analysis_prompt(filing_content, filing_metadata)

            # Call Claude API
            logger.info(f"Analyzing filing {filing_id} with {self.model}")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.MAX_OUTPUT_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Extract response text
            response_text = response.content[0].text

            # Parse JSON response
            try:
                analysis = json.loads(response_text)
            except json.JSONDecodeError as e:
                # Try to extract JSON from response
                logger.warning(f"Failed to parse JSON directly, attempting extraction: {e}")
                analysis = self._extract_json_from_response(response_text)
                if not analysis:
                    raise ValueError("Could not extract valid JSON from response")

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Build language patterns
            language_patterns = []
            for pattern in analysis.get("language_patterns", []):
                language_patterns.append(LanguagePattern(
                    pattern_type=pattern.get("pattern_type", "unknown"),
                    text_excerpt=pattern.get("text_excerpt", "")[:200],
                    confidence=float(pattern.get("confidence", 0.5))
                ))

            # Build result
            result = AIAnalysisResult(
                filing_id=filing_id,
                sentiment_score=float(analysis.get("sentiment_score", 0)),
                sentiment_category=SentimentCategory(
                    analysis.get("sentiment_category", "neutral")
                ),
                confidence=float(analysis.get("confidence", 0.5)),
                hedging_score=float(analysis.get("hedging_score", 0.5)),
                confidence_language_score=float(
                    analysis.get("confidence_language_score", 0.5)
                ),
                key_insights=analysis.get("key_insights", [])[:5],
                trial_mentions=analysis.get("trial_mentions", []),
                forward_looking_statements=analysis.get(
                    "forward_looking_statements", []
                )[:5],
                risk_factors=analysis.get("risk_factors", [])[:5],
                language_patterns=language_patterns,
                raw_analysis=analysis,
                processing_time_ms=processing_time_ms,
                model_used=self.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens
            )

            logger.info(
                f"Analysis complete for {filing_id}: "
                f"score={result.sentiment_score:.1f}, "
                f"category={result.sentiment_category.value}, "
                f"confidence={result.confidence:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to analyze filing {filing_id}: {e}")
            return None

    def _extract_json_from_response(self, text: str) -> Optional[Dict]:
        """
        Attempt to extract JSON from a response that may contain extra text.

        Args:
            text: Response text that may contain JSON

        Returns:
            Parsed JSON dict or None
        """
        # Try to find JSON object in response
        import re

        # Look for JSON object pattern
        json_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_pattern, text)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    def score_to_signal_weight(self, sentiment_score: float) -> int:
        """
        Convert sentiment score (-10 to +10) to signal weight for scoring model.

        Args:
            sentiment_score: Score from -10 to +10

        Returns:
            Signal weight (positive for bullish, negative for bearish)
        """
        # Map sentiment score to weight categories
        if sentiment_score >= 8:
            return 5  # Very bullish
        elif sentiment_score >= 5:
            return 4  # Bullish
        elif sentiment_score >= 2:
            return 2  # Mildly bullish
        elif sentiment_score >= -2:
            return 0  # Neutral
        elif sentiment_score >= -5:
            return -2  # Mildly bearish
        elif sentiment_score >= -8:
            return -4  # Bearish
        else:
            return -5  # Very bearish

    def analyze_and_create_signal(
        self,
        filing_content: str,
        filing_metadata: Dict,
        trial_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a filing and create a signal dict ready for database insertion.

        Args:
            filing_content: Raw filing content
            filing_metadata: Filing metadata
            trial_id: Optional trial ID to associate with signal

        Returns:
            Signal dict ready for TrialSignal creation, or None
        """
        result = self.analyze_filing(filing_content, filing_metadata)

        if not result:
            return None

        # Determine signal type based on sentiment
        if result.sentiment_score > 0:
            signal_type = "sec_ai_positive"
        elif result.sentiment_score < 0:
            signal_type = "sec_ai_negative"
        else:
            signal_type = "sec_ai_neutral"

        # Build signal value (summary)
        signal_value = (
            f"AI Analysis: {result.sentiment_category.value} "
            f"(score: {result.sentiment_score:.1f}/10, "
            f"confidence: {result.confidence:.0%}). "
        )

        if result.key_insights:
            signal_value += f"Key insight: {result.key_insights[0]}"

        signal = {
            "trial_id": trial_id or filing_metadata.get("trial_id"),
            "signal_type": signal_type,
            "signal_value": signal_value[:500],  # Truncate if needed
            "signal_weight": self.score_to_signal_weight(result.sentiment_score),
            "source": "sec_ai_analysis",
            "source_url": filing_metadata.get("filing_url"),
            "raw_data": {
                "sentiment_score": result.sentiment_score,
                "sentiment_category": result.sentiment_category.value,
                "confidence": result.confidence,
                "hedging_score": result.hedging_score,
                "confidence_language_score": result.confidence_language_score,
                "key_insights": result.key_insights,
                "trial_mentions": result.trial_mentions,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "processing_time_ms": result.processing_time_ms,
            }
        }

        return signal

    def batch_analyze_filings(
        self,
        filings: List[Tuple[str, Dict]],
        max_concurrent: int = 1
    ) -> List[AIAnalysisResult]:
        """
        Analyze multiple filings.

        Args:
            filings: List of (content, metadata) tuples
            max_concurrent: Max concurrent requests (default 1 for rate limiting)

        Returns:
            List of analysis results
        """
        results = []

        for i, (content, metadata) in enumerate(filings):
            logger.info(f"Analyzing filing {i+1}/{len(filings)}")

            result = self.analyze_filing(
                content,
                metadata,
                filing_id=metadata.get("accession_number", f"filing_{i}")
            )

            if result:
                results.append(result)

        logger.info(f"Batch analysis complete: {len(results)}/{len(filings)} successful")
        return results


class MockSECAIAnalyzer(SECAIAnalyzer):
    """
    Mock analyzer for testing without API calls.

    Returns deterministic results based on filing content keywords.
    """

    def __init__(self):
        """Initialize mock analyzer (no API key needed)."""
        self.api_key = "mock_key"
        self.model = "mock-model"
        self._last_request_time = 0

    def analyze_filing(
        self,
        filing_content: str,
        filing_metadata: Dict,
        filing_id: Optional[str] = None
    ) -> Optional[AIAnalysisResult]:
        """
        Generate mock analysis based on keyword detection.

        Args:
            filing_content: Filing content
            filing_metadata: Filing metadata
            filing_id: Optional filing ID

        Returns:
            Mock AIAnalysisResult
        """
        content_lower = filing_content.lower()

        # Detect sentiment based on keywords
        positive_keywords = [
            "positive", "met primary endpoint", "statistically significant",
            "exceeded", "breakthrough", "accelerated", "strong efficacy"
        ]
        negative_keywords = [
            "failed", "did not meet", "terminated", "discontinued",
            "safety concern", "adverse event", "clinical hold"
        ]

        positive_count = sum(1 for kw in positive_keywords if kw in content_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in content_lower)

        # Calculate mock sentiment score
        if positive_count > negative_count:
            sentiment_score = min(2.0 + positive_count * 1.5, 10.0)
            category = SentimentCategory.BULLISH if sentiment_score < 7 else SentimentCategory.VERY_BULLISH
        elif negative_count > positive_count:
            sentiment_score = max(-2.0 - negative_count * 1.5, -10.0)
            category = SentimentCategory.BEARISH if sentiment_score > -7 else SentimentCategory.VERY_BEARISH
        else:
            sentiment_score = 0.0
            category = SentimentCategory.NEUTRAL

        # Detect hedging language
        hedging_words = ["may", "might", "could", "potentially", "uncertain", "no assurance"]
        confident_words = ["will", "expect", "confident", "on track", "committed"]

        hedging_count = sum(1 for w in hedging_words if w in content_lower)
        confident_count = sum(1 for w in confident_words if w in content_lower)

        total_lang = hedging_count + confident_count + 1
        hedging_score = hedging_count / total_lang
        confidence_language_score = confident_count / total_lang

        return AIAnalysisResult(
            filing_id=filing_id or "mock_filing",
            sentiment_score=sentiment_score,
            sentiment_category=category,
            confidence=0.75,
            hedging_score=hedging_score,
            confidence_language_score=confidence_language_score,
            key_insights=["Mock analysis - keyword-based sentiment detection"],
            trial_mentions=[],
            forward_looking_statements=[],
            risk_factors=[],
            language_patterns=[],
            raw_analysis={"mock": True},
            processing_time_ms=50,
            model_used="mock-model",
            tokens_used=0
        )


# Export for use in config
def get_analyzer(use_mock: bool = False) -> SECAIAnalyzer:
    """
    Factory function to get appropriate analyzer.

    Args:
        use_mock: If True, return mock analyzer for testing

    Returns:
        SECAIAnalyzer instance
    """
    if use_mock or not os.getenv("ANTHROPIC_API_KEY"):
        logger.info("Using mock SEC AI analyzer")
        return MockSECAIAnalyzer()
    return SECAIAnalyzer()


if __name__ == "__main__":
    # Test the analyzer
    logging.basicConfig(level=logging.INFO)

    print("Testing SEC AI Analyzer...")
    print("=" * 60)

    # Test with mock analyzer
    mock_analyzer = MockSECAIAnalyzer()

    # Sample filing content
    sample_content = """
    UNITED STATES SECURITIES AND EXCHANGE COMMISSION
    Washington, D.C. 20549
    FORM 8-K

    Item 8.01 Other Events

    On November 15, 2024, BioTech Corp announced positive topline results from its
    Phase 3 clinical trial evaluating Drug-123 for the treatment of advanced cancer.

    The trial met its primary endpoint, demonstrating statistically significant
    improvement in overall survival compared to standard of care (HR=0.72, p<0.001).

    The company is confident in these results and expects to submit a New Drug
    Application to the FDA in Q1 2025. Management believes this breakthrough therapy
    designation will support an accelerated review timeline.

    Forward-Looking Statements:
    This report contains forward-looking statements that are subject to risks and
    uncertainties. Actual results may differ materially from those projected.
    """

    sample_metadata = {
        "company_name": "BioTech Corp",
        "ticker": "BTCH",
        "filing_type": "8-K",
        "filing_date": "2024-11-15",
        "accession_number": "0001234567-24-000001"
    }

    print("\nAnalyzing sample positive filing...")
    result = mock_analyzer.analyze_filing(sample_content, sample_metadata)

    if result:
        print(f"\nResults:")
        print(f"  Sentiment Score: {result.sentiment_score:.1f}/10")
        print(f"  Category: {result.sentiment_category.value}")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Hedging Score: {result.hedging_score:.2f}")
        print(f"  Confidence Language: {result.confidence_language_score:.2f}")
        print(f"  Signal Weight: {mock_analyzer.score_to_signal_weight(result.sentiment_score)}")

    # Test negative content
    negative_content = """
    FORM 8-K

    Item 8.01 Other Events

    BioTech Corp today announced that its Phase 3 clinical trial for Drug-456
    did not meet its primary endpoint. The trial was terminated due to safety
    concerns and adverse events reported in the treatment arm.

    The company has discontinued development of this program. There is no
    assurance that other pipeline programs will succeed.
    """

    print("\n" + "=" * 60)
    print("Analyzing sample negative filing...")
    result = mock_analyzer.analyze_filing(negative_content, sample_metadata)

    if result:
        print(f"\nResults:")
        print(f"  Sentiment Score: {result.sentiment_score:.1f}/10")
        print(f"  Category: {result.sentiment_category.value}")
        print(f"  Signal Weight: {mock_analyzer.score_to_signal_weight(result.sentiment_score)}")

    print("\n" + "=" * 60)
    print("Testing signal creation...")
    signal = mock_analyzer.analyze_and_create_signal(
        sample_content,
        sample_metadata,
        trial_id="NCT12345678"
    )

    if signal:
        print(f"\nGenerated Signal:")
        print(f"  Type: {signal['signal_type']}")
        print(f"  Weight: {signal['signal_weight']}")
        print(f"  Value: {signal['signal_value'][:100]}...")

    print("\nTest complete!")
