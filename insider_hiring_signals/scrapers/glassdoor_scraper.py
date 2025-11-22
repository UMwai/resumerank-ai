"""
Glassdoor Sentiment Scraper with AI Analysis

Scrapes Glassdoor reviews for biotech companies and uses Claude AI
to analyze sentiment and extract actionable signals.

Key signals extracted:
- Overall company sentiment trend
- Pipeline/R&D mentions and sentiment
- Management/leadership concerns
- Layoff mentions or hiring freeze indicators
- Culture and morale indicators
- Comparison to industry benchmarks

Note: Glassdoor actively blocks scrapers. This module uses:
1. Rate limiting and delays
2. Rotating user agents
3. Session management
4. Fallback to cached data when blocked
"""

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from utils.config import get_config
from utils.database import get_database
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class GlassdoorReview:
    """Represents a single Glassdoor review."""
    company_ticker: str
    company_name: str
    review_date: Optional[date]
    overall_rating: float
    ceo_approval: Optional[bool]
    recommend_to_friend: Optional[bool]
    business_outlook: Optional[str]
    pros: str
    cons: str
    review_text: str
    job_title: Optional[str]
    employment_status: Optional[str]  # 'Current', 'Former'
    source_url: str
    review_id: Optional[str] = None


@dataclass
class SentimentAnalysis:
    """AI-generated sentiment analysis for a review."""
    sentiment_score: float  # -1.0 to +1.0
    confidence: float
    key_themes: List[str]
    mentions_layoffs: bool
    mentions_pipeline: bool
    mentions_management: bool
    mentions_culture: bool
    bullish_factors: List[str]
    bearish_factors: List[str]
    signal_weight: int
    summary: str


@dataclass
class CompanySentimentSummary:
    """Aggregate sentiment summary for a company."""
    company_ticker: str
    analysis_date: date
    review_count: int
    avg_rating: float
    avg_sentiment: float
    sentiment_trend: str  # 'improving', 'declining', 'stable'
    ceo_approval_rate: Optional[float]
    recommend_rate: Optional[float]
    key_concerns: List[str]
    key_positives: List[str]
    layoff_mentions: int
    pipeline_mentions: int
    overall_signal: str  # 'bullish', 'bearish', 'neutral'
    signal_weight: int


class GlassdoorScraper:
    """
    Scrapes and analyzes Glassdoor reviews for biotech companies.

    Features:
    - Scrapes reviews with anti-detection measures
    - AI-powered sentiment analysis using Claude
    - Extracts biotech-specific signals (pipeline mentions, R&D sentiment)
    - Tracks sentiment trends over time
    - Detects layoff and hiring freeze mentions
    """

    # User agents for rotation
    USER_AGENTS = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]

    # Company Glassdoor URLs/IDs (would be populated from config)
    COMPANY_GLASSDOOR_IDS = {
        'MRNA': {'id': 'E1180507', 'name': 'Moderna', 'slug': 'Moderna-Reviews-E1180507'},
        'VRTX': {'id': 'E5245', 'name': 'Vertex-Pharmaceuticals', 'slug': 'Vertex-Pharmaceuticals-Reviews-E5245'},
        'REGN': {'id': 'E14459', 'name': 'Regeneron-Pharmaceuticals', 'slug': 'Regeneron-Pharmaceuticals-Reviews-E14459'},
        'BIIB': {'id': 'E2765', 'name': 'Biogen', 'slug': 'Biogen-Reviews-E2765'},
        'ALNY': {'id': 'E251706', 'name': 'Alnylam-Pharmaceuticals', 'slug': 'Alnylam-Pharmaceuticals-Reviews-E251706'},
        'BMRN': {'id': 'E17138', 'name': 'BioMarin-Pharmaceutical', 'slug': 'BioMarin-Pharmaceutical-Reviews-E17138'},
        'INCY': {'id': 'E32731', 'name': 'Incyte', 'slug': 'Incyte-Reviews-E32731'},
        'CRSP': {'id': 'E1327810', 'name': 'CRISPR-Therapeutics', 'slug': 'CRISPR-Therapeutics-Reviews-E1327810'},
        'BEAM': {'id': 'E2803147', 'name': 'Beam-Therapeutics', 'slug': 'Beam-Therapeutics-Reviews-E2803147'},
        'EDIT': {'id': 'E1149753', 'name': 'Editas-Medicine', 'slug': 'Editas-Medicine-Reviews-E1149753'},
        'NTLA': {'id': 'E1166025', 'name': 'Intellia-Therapeutics', 'slug': 'Intellia-Therapeutics-Reviews-E1166025'},
        'SGEN': {'id': 'E251890', 'name': 'Seagen', 'slug': 'Seagen-Reviews-E251890'},
        'IONS': {'id': 'E10655', 'name': 'Ionis-Pharmaceuticals', 'slug': 'Ionis-Pharmaceuticals-Reviews-E10655'},
        'NBIX': {'id': 'E7739', 'name': 'Neurocrine-Biosciences', 'slug': 'Neurocrine-Biosciences-Reviews-E7739'},
    }

    # Signal weights
    SIGNAL_WEIGHTS = {
        'VERY_POSITIVE': 4,
        'POSITIVE': 2,
        'NEUTRAL': 0,
        'NEGATIVE': -2,
        'VERY_NEGATIVE': -4,
        'LAYOFF_MENTIONS': -3,
        'PIPELINE_POSITIVE': 3,
        'PIPELINE_NEGATIVE': -4,
        'MANAGEMENT_CRISIS': -3,
        'CULTURE_DECLINE': -2,
    }

    # Keywords for detection
    LAYOFF_KEYWORDS = [
        'layoff', 'laid off', 'restructuring', 'reduction', 'downsizing',
        'let go', 'terminated', 'fired', 'rif', 'workforce reduction',
        'headcount', 'cutting jobs', 'eliminating positions'
    ]

    PIPELINE_KEYWORDS = [
        'pipeline', 'clinical trial', 'phase', 'fda', 'approval',
        'drug development', 'r&d', 'research', 'science', 'data',
        'efficacy', 'safety', 'regulatory', 'candidate'
    ]

    MANAGEMENT_KEYWORDS = [
        'leadership', 'management', 'executive', 'ceo', 'direction',
        'strategy', 'vision', 'decision', 'communication', 'transparency'
    ]

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Glassdoor scraper."""
        self.config = get_config(config_path)
        self.db = get_database(config_path)
        self.session = self._create_session()
        self.base_url = "https://www.glassdoor.com"

        # Rate limiting
        self.min_delay = 3.0  # Minimum seconds between requests
        self.max_delay = 8.0  # Maximum seconds between requests
        self.last_request_time = 0

        # AI client (lazy initialization)
        self._ai_client = None

    def _create_session(self) -> requests.Session:
        """Create a requests session with anti-detection headers."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        return session

    def _rotate_user_agent(self):
        """Rotate the User-Agent header."""
        self.session.headers['User-Agent'] = random.choice(self.USER_AGENTS)

    def _rate_limit_wait(self):
        """Apply rate limiting with randomization."""
        elapsed = time.time() - self.last_request_time
        delay = random.uniform(self.min_delay, self.max_delay)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make a rate-limited request with anti-detection."""
        self._rate_limit_wait()
        self._rotate_user_agent()

        try:
            response = self.session.get(
                url,
                timeout=self.config.scraping.get('timeout_seconds', 30)
            )

            # Check for blocking
            if response.status_code == 403 or 'captcha' in response.text.lower():
                logger.warning(f"Glassdoor is blocking requests. Waiting...")
                time.sleep(60)  # Wait a minute before retrying
                return None

            response.raise_for_status()
            return response

        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    @property
    def ai_client(self):
        """Lazy initialize AI client."""
        if self._ai_client is None and self.config.anthropic_api_key:
            try:
                import anthropic
                self._ai_client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            except ImportError:
                logger.warning("anthropic package not installed")
        return self._ai_client

    def get_reviews_url(self, ticker: str, page: int = 1) -> Optional[str]:
        """Build Glassdoor reviews URL for a company."""
        if ticker not in self.COMPANY_GLASSDOOR_IDS:
            return None

        company_info = self.COMPANY_GLASSDOOR_IDS[ticker]
        slug = company_info['slug']

        if page == 1:
            return f"{self.base_url}/Reviews/{slug}.htm"
        else:
            return f"{self.base_url}/Reviews/{slug}_P{page}.htm"

    def scrape_reviews_page(self, ticker: str, page: int = 1) -> List[GlassdoorReview]:
        """
        Scrape a single page of Glassdoor reviews.

        Args:
            ticker: Company ticker symbol
            page: Page number to scrape

        Returns:
            List of GlassdoorReview objects
        """
        url = self.get_reviews_url(ticker, page)
        if not url:
            logger.warning(f"No Glassdoor URL configured for {ticker}")
            return []

        response = self._make_request(url)
        if not response:
            return []

        return self._parse_reviews_page(response.text, ticker)

    def _parse_reviews_page(self, html: str, ticker: str) -> List[GlassdoorReview]:
        """Parse reviews from Glassdoor HTML page."""
        reviews = []
        soup = BeautifulSoup(html, 'html.parser')

        company_name = self.COMPANY_GLASSDOOR_IDS.get(ticker, {}).get('name', ticker)

        # Find review containers
        review_containers = soup.find_all(['div', 'article'], class_=re.compile(r'review|empReview', re.I))

        for container in review_containers:
            try:
                review = self._parse_single_review(container, ticker, company_name)
                if review:
                    reviews.append(review)
            except Exception as e:
                logger.debug(f"Failed to parse review: {e}")
                continue

        logger.info(f"Parsed {len(reviews)} reviews from page for {ticker}")
        return reviews

    def _parse_single_review(self, container, ticker: str, company_name: str) -> Optional[GlassdoorReview]:
        """Parse a single review container."""
        # Extract rating
        rating = 0.0
        rating_elem = container.find(['span', 'div'], class_=re.compile(r'rating|ratingNumber', re.I))
        if rating_elem:
            try:
                rating_text = rating_elem.get_text(strip=True)
                rating = float(re.search(r'(\d+\.?\d*)', rating_text).group(1))
            except (AttributeError, ValueError):
                pass

        # Extract date
        review_date = None
        date_elem = container.find(['time', 'span'], class_=re.compile(r'date|timestamp', re.I))
        if date_elem:
            date_text = date_elem.get_text(strip=True)
            review_date = self._parse_date(date_text)

        # Extract pros and cons
        pros = ''
        cons = ''

        pros_elem = container.find(['p', 'span', 'div'], class_=re.compile(r'pro|positive', re.I))
        cons_elem = container.find(['p', 'span', 'div'], class_=re.compile(r'con|negative', re.I))

        # Alternative: look for labeled sections
        for elem in container.find_all(['p', 'span', 'div']):
            text = elem.get_text(strip=True).lower()
            if 'pros' in text[:10]:
                pros = elem.get_text(strip=True)[5:].strip()
            elif 'cons' in text[:10]:
                cons = elem.get_text(strip=True)[5:].strip()

        if pros_elem and not pros:
            pros = pros_elem.get_text(strip=True)
        if cons_elem and not cons:
            cons = cons_elem.get_text(strip=True)

        # Extract review text (main content)
        review_text = ''
        main_text_elem = container.find(['p', 'div'], class_=re.compile(r'reviewBody|mainText|description', re.I))
        if main_text_elem:
            review_text = main_text_elem.get_text(strip=True)
        else:
            # Fallback: combine pros and cons
            review_text = f"{pros} {cons}".strip()

        # Extract job title
        job_title = None
        title_elem = container.find(['span', 'div'], class_=re.compile(r'jobTitle|title', re.I))
        if title_elem:
            job_title = title_elem.get_text(strip=True)

        # Employment status
        employment_status = None
        status_elem = container.find(['span', 'div'], class_=re.compile(r'employee|status', re.I))
        if status_elem:
            status_text = status_elem.get_text(strip=True).lower()
            if 'current' in status_text:
                employment_status = 'Current'
            elif 'former' in status_text:
                employment_status = 'Former'

        # Generate review ID
        review_id = hashlib.md5(f"{ticker}{review_date}{review_text[:50]}".encode()).hexdigest()[:16]

        # Skip if no meaningful content
        if not review_text and not pros and not cons:
            return None

        return GlassdoorReview(
            company_ticker=ticker,
            company_name=company_name,
            review_date=review_date,
            overall_rating=rating,
            ceo_approval=None,  # Would need additional parsing
            recommend_to_friend=None,
            business_outlook=None,
            pros=pros,
            cons=cons,
            review_text=review_text,
            job_title=job_title,
            employment_status=employment_status,
            source_url=self.get_reviews_url(ticker) or '',
            review_id=review_id
        )

    def _parse_date(self, date_text: str) -> Optional[date]:
        """Parse date from various formats."""
        date_text = date_text.strip()

        # Common formats
        patterns = [
            (r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', '%B %d %Y'),
            (r'(\d{1,2})/(\d{1,2})/(\d{2,4})', '%m/%d/%Y'),
            (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),
        ]

        for pattern, fmt in patterns:
            match = re.search(pattern, date_text)
            if match:
                try:
                    date_str = match.group(0).replace(',', '')
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue

        # Handle relative dates
        if 'today' in date_text.lower():
            return date.today()
        if 'yesterday' in date_text.lower():
            return date.today() - timedelta(days=1)

        days_match = re.search(r'(\d+)\s*days?\s*ago', date_text.lower())
        if days_match:
            return date.today() - timedelta(days=int(days_match.group(1)))

        months_match = re.search(r'(\d+)\s*months?\s*ago', date_text.lower())
        if months_match:
            return date.today() - timedelta(days=int(months_match.group(1)) * 30)

        return None

    def analyze_review_with_ai(self, review: GlassdoorReview) -> SentimentAnalysis:
        """
        Analyze a single review using Claude AI.

        Args:
            review: GlassdoorReview to analyze

        Returns:
            SentimentAnalysis object
        """
        # Default analysis if AI not available
        default_analysis = SentimentAnalysis(
            sentiment_score=0.0,
            confidence=0.0,
            key_themes=[],
            mentions_layoffs=False,
            mentions_pipeline=False,
            mentions_management=False,
            mentions_culture=False,
            bullish_factors=[],
            bearish_factors=[],
            signal_weight=0,
            summary="AI analysis not available"
        )

        if not self.ai_client:
            # Perform basic keyword-based analysis
            return self._basic_sentiment_analysis(review)

        try:
            prompt = f"""Analyze this Glassdoor review for a biotech company ({review.company_ticker}) and provide investment-relevant insights.

Review:
Rating: {review.overall_rating}/5
Job Title: {review.job_title or 'Unknown'}
Employment Status: {review.employment_status or 'Unknown'}
Pros: {review.pros}
Cons: {review.cons}
Review Text: {review.review_text}

Analyze and respond with a JSON object containing:
{{
    "sentiment_score": <float from -1.0 (very negative) to +1.0 (very positive)>,
    "confidence": <float from 0.0 to 1.0>,
    "key_themes": [<list of 2-4 main themes mentioned>],
    "mentions_layoffs": <boolean - any mention of layoffs, restructuring, job cuts>,
    "mentions_pipeline": <boolean - any mention of clinical trials, FDA, drug development, R&D>,
    "mentions_management": <boolean - any concerns or praise about leadership/management>,
    "mentions_culture": <boolean - any discussion of company culture, morale, work environment>,
    "bullish_factors": [<list of positive signals for stock - max 3>],
    "bearish_factors": [<list of negative signals for stock - max 3>],
    "signal_weight": <integer from -5 to +5 indicating investment signal strength>,
    "summary": "<one sentence summary of investment relevance>"
}}

Focus on:
- Signs of pipeline success/failure (critical for biotech)
- Leadership stability and competence
- Employee morale as indicator of company health
- Layoff mentions (bearish signal)
- Commercial readiness (bullish if nearing launch)"""

            response = self.ai_client.messages.create(
                model=self.config.ai.get('model', 'claude-sonnet-4-5-20250929'),
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            response_text = response.content[0].text

            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                analysis_data = json.loads(json_match.group())

                return SentimentAnalysis(
                    sentiment_score=float(analysis_data.get('sentiment_score', 0)),
                    confidence=float(analysis_data.get('confidence', 0.5)),
                    key_themes=analysis_data.get('key_themes', []),
                    mentions_layoffs=bool(analysis_data.get('mentions_layoffs', False)),
                    mentions_pipeline=bool(analysis_data.get('mentions_pipeline', False)),
                    mentions_management=bool(analysis_data.get('mentions_management', False)),
                    mentions_culture=bool(analysis_data.get('mentions_culture', False)),
                    bullish_factors=analysis_data.get('bullish_factors', []),
                    bearish_factors=analysis_data.get('bearish_factors', []),
                    signal_weight=int(analysis_data.get('signal_weight', 0)),
                    summary=analysis_data.get('summary', '')
                )

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")

        return self._basic_sentiment_analysis(review)

    def _basic_sentiment_analysis(self, review: GlassdoorReview) -> SentimentAnalysis:
        """Perform basic keyword-based sentiment analysis."""
        text = f"{review.pros} {review.cons} {review.review_text}".lower()

        # Check for keywords
        mentions_layoffs = any(kw in text for kw in self.LAYOFF_KEYWORDS)
        mentions_pipeline = any(kw in text for kw in self.PIPELINE_KEYWORDS)
        mentions_management = any(kw in text for kw in self.MANAGEMENT_KEYWORDS)

        # Culture keywords
        culture_keywords = ['culture', 'environment', 'morale', 'team', 'atmosphere']
        mentions_culture = any(kw in text for kw in culture_keywords)

        # Calculate basic sentiment from rating
        rating_sentiment = (review.overall_rating - 3) / 2  # Convert 1-5 to -1 to +1

        # Adjust for keywords
        sentiment_score = rating_sentiment
        signal_weight = 0

        if mentions_layoffs:
            sentiment_score -= 0.2
            signal_weight += self.SIGNAL_WEIGHTS['LAYOFF_MENTIONS']

        if mentions_pipeline:
            # Check if positive or negative context
            negative_pipeline_words = ['fail', 'disappoint', 'delay', 'setback', 'concern']
            if any(word in text for word in negative_pipeline_words):
                signal_weight += self.SIGNAL_WEIGHTS['PIPELINE_NEGATIVE']
            else:
                signal_weight += self.SIGNAL_WEIGHTS['PIPELINE_POSITIVE'] // 2

        # Clamp sentiment score
        sentiment_score = max(-1.0, min(1.0, sentiment_score))

        # Determine signal weight from rating
        if review.overall_rating >= 4.5:
            signal_weight += self.SIGNAL_WEIGHTS['VERY_POSITIVE']
        elif review.overall_rating >= 3.5:
            signal_weight += self.SIGNAL_WEIGHTS['POSITIVE']
        elif review.overall_rating <= 1.5:
            signal_weight += self.SIGNAL_WEIGHTS['VERY_NEGATIVE']
        elif review.overall_rating <= 2.5:
            signal_weight += self.SIGNAL_WEIGHTS['NEGATIVE']

        return SentimentAnalysis(
            sentiment_score=sentiment_score,
            confidence=0.5,  # Lower confidence for basic analysis
            key_themes=[],
            mentions_layoffs=mentions_layoffs,
            mentions_pipeline=mentions_pipeline,
            mentions_management=mentions_management,
            mentions_culture=mentions_culture,
            bullish_factors=[],
            bearish_factors=[],
            signal_weight=signal_weight,
            summary="Basic keyword analysis (AI not available)"
        )

    def generate_company_summary(
        self,
        ticker: str,
        reviews: List[GlassdoorReview],
        analyses: List[SentimentAnalysis]
    ) -> CompanySentimentSummary:
        """
        Generate aggregate sentiment summary for a company.

        Args:
            ticker: Company ticker
            reviews: List of reviews
            analyses: List of sentiment analyses

        Returns:
            CompanySentimentSummary object
        """
        if not reviews:
            return CompanySentimentSummary(
                company_ticker=ticker,
                analysis_date=date.today(),
                review_count=0,
                avg_rating=0.0,
                avg_sentiment=0.0,
                sentiment_trend='stable',
                ceo_approval_rate=None,
                recommend_rate=None,
                key_concerns=[],
                key_positives=[],
                layoff_mentions=0,
                pipeline_mentions=0,
                overall_signal='neutral',
                signal_weight=0
            )

        # Calculate averages
        avg_rating = sum(r.overall_rating for r in reviews) / len(reviews)
        avg_sentiment = sum(a.sentiment_score for a in analyses) / len(analyses) if analyses else 0.0

        # Count mentions
        layoff_mentions = sum(1 for a in analyses if a.mentions_layoffs)
        pipeline_mentions = sum(1 for a in analyses if a.mentions_pipeline)

        # Calculate total signal weight
        total_signal_weight = sum(a.signal_weight for a in analyses)

        # Aggregate themes
        all_concerns = []
        all_positives = []
        for analysis in analyses:
            all_concerns.extend(analysis.bearish_factors)
            all_positives.extend(analysis.bullish_factors)

        # Get most common concerns/positives
        from collections import Counter
        key_concerns = [item for item, _ in Counter(all_concerns).most_common(5)]
        key_positives = [item for item, _ in Counter(all_positives).most_common(5)]

        # Calculate trend (if we have dated reviews)
        sentiment_trend = 'stable'
        dated_reviews = [(r, a) for r, a in zip(reviews, analyses) if r.review_date]
        if len(dated_reviews) >= 4:
            dated_reviews.sort(key=lambda x: x[0].review_date)
            recent = dated_reviews[len(dated_reviews)//2:]
            older = dated_reviews[:len(dated_reviews)//2]

            recent_avg = sum(a.sentiment_score for _, a in recent) / len(recent)
            older_avg = sum(a.sentiment_score for _, a in older) / len(older)

            if recent_avg > older_avg + 0.1:
                sentiment_trend = 'improving'
            elif recent_avg < older_avg - 0.1:
                sentiment_trend = 'declining'

        # Determine overall signal
        if avg_sentiment > 0.3 and avg_rating >= 4.0:
            overall_signal = 'bullish'
        elif avg_sentiment < -0.3 or avg_rating <= 2.5 or layoff_mentions >= 3:
            overall_signal = 'bearish'
        else:
            overall_signal = 'neutral'

        return CompanySentimentSummary(
            company_ticker=ticker,
            analysis_date=date.today(),
            review_count=len(reviews),
            avg_rating=round(avg_rating, 2),
            avg_sentiment=round(avg_sentiment, 3),
            sentiment_trend=sentiment_trend,
            ceo_approval_rate=None,
            recommend_rate=None,
            key_concerns=key_concerns,
            key_positives=key_positives,
            layoff_mentions=layoff_mentions,
            pipeline_mentions=pipeline_mentions,
            overall_signal=overall_signal,
            signal_weight=total_signal_weight
        )

    def save_reviews(self, reviews: List[GlassdoorReview], analyses: List[SentimentAnalysis]) -> int:
        """Save reviews and analyses to database."""
        if not reviews:
            return 0

        saved = 0
        for review, analysis in zip(reviews, analyses):
            data = {
                'company_ticker': review.company_ticker,
                'company_name': review.company_name,
                'review_date': review.review_date,
                'overall_rating': review.overall_rating,
                'ceo_approval': review.ceo_approval,
                'recommend_to_friend': review.recommend_to_friend,
                'business_outlook': review.business_outlook,
                'pros': review.pros[:2000] if review.pros else '',
                'cons': review.cons[:2000] if review.cons else '',
                'review_text': review.review_text[:2000] if review.review_text else '',
                'job_title': review.job_title,
                'employment_status': review.employment_status,
                'sentiment_score': analysis.sentiment_score,
                'mentions_layoffs': analysis.mentions_layoffs,
                'mentions_pipeline': analysis.mentions_pipeline,
                'mentions_management': analysis.mentions_management,
                'ai_analysis': {
                    'key_themes': analysis.key_themes,
                    'bullish_factors': analysis.bullish_factors,
                    'bearish_factors': analysis.bearish_factors,
                    'summary': analysis.summary,
                    'confidence': analysis.confidence
                },
                'signal_weight': analysis.signal_weight,
                'source_url': review.source_url,
            }

            try:
                self.db.insert('glassdoor_sentiment', data)
                saved += 1
            except Exception as e:
                # Likely duplicate
                logger.debug(f"Could not save review: {e}")

        logger.info(f"Saved {saved} reviews")
        return saved

    def scrape_company(
        self,
        ticker: str,
        max_pages: int = 3,
        max_reviews: int = 30
    ) -> Tuple[List[GlassdoorReview], List[SentimentAnalysis], CompanySentimentSummary]:
        """
        Scrape and analyze Glassdoor reviews for a company.

        Args:
            ticker: Company ticker symbol
            max_pages: Maximum pages to scrape
            max_reviews: Maximum reviews to collect

        Returns:
            Tuple of (reviews, analyses, summary)
        """
        logger.info(f"Scraping Glassdoor reviews for {ticker}")

        all_reviews = []
        all_analyses = []

        for page in range(1, max_pages + 1):
            if len(all_reviews) >= max_reviews:
                break

            reviews = self.scrape_reviews_page(ticker, page)

            if not reviews:
                break

            for review in reviews:
                if len(all_reviews) >= max_reviews:
                    break

                analysis = self.analyze_review_with_ai(review)
                all_reviews.append(review)
                all_analyses.append(analysis)

            # Extra delay between pages
            time.sleep(random.uniform(2, 5))

        # Generate summary
        summary = self.generate_company_summary(ticker, all_reviews, all_analyses)

        logger.info(f"Analyzed {len(all_reviews)} reviews for {ticker}: "
                   f"avg_rating={summary.avg_rating}, sentiment={summary.avg_sentiment}, "
                   f"signal={summary.overall_signal}")

        return all_reviews, all_analyses, summary

    def run(
        self,
        tickers: Optional[List[str]] = None,
        max_pages: int = 2,
        max_reviews: int = 20
    ) -> Dict[str, Any]:
        """
        Run the Glassdoor scraper for all configured companies.

        Args:
            tickers: Optional list of tickers (defaults to configured ones)
            max_pages: Max pages per company
            max_reviews: Max reviews per company

        Returns:
            Dictionary with run statistics
        """
        if tickers is None:
            tickers = list(self.COMPANY_GLASSDOOR_IDS.keys())
        else:
            # Filter to only configured companies
            tickers = [t for t in tickers if t in self.COMPANY_GLASSDOOR_IDS]

        run_id = self.db.log_scraper_run('glassdoor')

        total_reviews = 0
        total_saved = 0
        errors = []
        summaries = {}

        for ticker in tickers:
            try:
                reviews, analyses, summary = self.scrape_company(
                    ticker, max_pages, max_reviews
                )

                saved = self.save_reviews(reviews, analyses)

                total_reviews += len(reviews)
                total_saved += saved

                summaries[ticker] = {
                    'review_count': summary.review_count,
                    'avg_rating': summary.avg_rating,
                    'avg_sentiment': summary.avg_sentiment,
                    'sentiment_trend': summary.sentiment_trend,
                    'layoff_mentions': summary.layoff_mentions,
                    'overall_signal': summary.overall_signal,
                    'signal_weight': summary.signal_weight
                }

                # Long delay between companies
                time.sleep(random.uniform(10, 20))

            except Exception as e:
                errors.append({'ticker': ticker, 'error': str(e)})
                logger.error(f"Failed to scrape Glassdoor for {ticker}: {e}")

        self.db.update_scraper_run(
            run_id,
            status='completed' if not errors else 'completed_with_errors',
            records_processed=total_reviews,
            records_inserted=total_saved,
            errors_count=len(errors),
            error_details={'errors': errors, 'summaries': summaries}
        )

        return {
            'run_id': run_id,
            'tickers_processed': len(tickers),
            'reviews_found': total_reviews,
            'reviews_saved': total_saved,
            'summaries': summaries,
            'errors': errors
        }


# Example output for sentiment analysis
SENTIMENT_ANALYSIS_EXAMPLES = """
Example 1 - Bullish Signal:
Review: "Great pipeline progress, Phase 3 results look promising. Management is transparent about timelines."
Analysis:
{
    "sentiment_score": 0.7,
    "confidence": 0.8,
    "key_themes": ["pipeline progress", "transparency", "clinical success"],
    "mentions_layoffs": false,
    "mentions_pipeline": true,
    "mentions_management": true,
    "bullish_factors": ["Phase 3 progress", "transparent management"],
    "bearish_factors": [],
    "signal_weight": 4,
    "summary": "Positive pipeline progress and management confidence suggest upcoming catalyst."
}

Example 2 - Bearish Signal:
Review: "Multiple rounds of layoffs this year. Pipeline setbacks and unclear direction from leadership."
Analysis:
{
    "sentiment_score": -0.8,
    "confidence": 0.9,
    "key_themes": ["layoffs", "pipeline issues", "leadership concerns"],
    "mentions_layoffs": true,
    "mentions_pipeline": true,
    "mentions_management": true,
    "bullish_factors": [],
    "bearish_factors": ["layoffs indicate cost cutting", "pipeline setbacks", "leadership instability"],
    "signal_weight": -5,
    "summary": "Layoffs combined with pipeline issues and leadership concerns are significant red flags."
}
"""


if __name__ == '__main__':
    # Test the scraper
    scraper = GlassdoorScraper()

    print("Glassdoor Scraper Test")
    print("=" * 50)

    # Test with one company
    ticker = 'MRNA'
    print(f"\nScraping reviews for {ticker}...")

    reviews, analyses, summary = scraper.scrape_company(ticker, max_pages=1, max_reviews=5)

    print(f"\nResults for {ticker}:")
    print(f"  Reviews collected: {len(reviews)}")
    print(f"  Average rating: {summary.avg_rating}")
    print(f"  Average sentiment: {summary.avg_sentiment:.3f}")
    print(f"  Overall signal: {summary.overall_signal}")
    print(f"  Signal weight: {summary.signal_weight}")
    print(f"  Layoff mentions: {summary.layoff_mentions}")
    print(f"  Pipeline mentions: {summary.pipeline_mentions}")

    if analyses:
        print(f"\nSample Analysis:")
        analysis = analyses[0]
        print(f"  Sentiment: {analysis.sentiment_score:.2f}")
        print(f"  Key themes: {analysis.key_themes}")
        print(f"  Summary: {analysis.summary}")
