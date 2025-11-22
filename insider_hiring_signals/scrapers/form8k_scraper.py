"""
SEC Form 8-K Executive Changes Scraper

Monitors SEC Form 8-K filings for Item 5.02 - Departure of Directors or Certain Officers;
Election of Directors; Appointment of Certain Officers; Compensatory Arrangements.

Key signals:
- C-suite departures (bearish, especially if unexpected)
- CMO/CSO departures in biotech (very bearish for pipeline)
- Interim appointments (uncertainty)
- Board member departures
- Mass executive exodus (highly bearish)

The SEC requires companies to file Form 8-K within 4 business days of:
- Director departures/elections
- Officer appointments/departures
- Principal officer changes (CEO, CFO, COO, etc.)
"""

import re
import time
import xml.etree.ElementTree as ET
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
class ExecutiveChange:
    """Represents an executive change from Form 8-K."""
    company_ticker: str
    company_cik: str
    executive_name: str
    title: str
    change_type: str  # 'Departure', 'Appointment', 'Promotion', 'Resignation', 'Termination'
    effective_date: Optional[date]
    announcement_date: date
    reason: Optional[str]
    is_voluntary: Optional[bool]
    successor_name: Optional[str]
    filing_url: str
    filing_text: str
    ai_analysis: Optional[Dict] = None
    severity_score: int = 0  # 1-10 scale
    signal_weight: int = 0


@dataclass
class Form8KFiling:
    """Represents a Form 8-K filing."""
    accession_number: str
    company_name: str
    company_cik: str
    filing_date: date
    filing_url: str
    items: List[str]
    has_item_502: bool = False


class Form8KScraper:
    """
    Scrapes SEC Form 8-K filings for executive changes (Item 5.02).

    The scraper:
    1. Fetches recent 8-K filings for watchlist companies
    2. Filters for Item 5.02 (executive changes)
    3. Parses the filing text to extract change details
    4. Optionally uses AI to analyze departure reasons
    5. Assigns signal weights based on executive role and change type
    """

    BASE_URL = "https://www.sec.gov"

    # Item 5.02 keywords
    ITEM_502_PATTERNS = [
        r'item\s*5\.02',
        r'departure\s+of\s+directors',
        r'departure\s+of\s+.*officers',
        r'election\s+of\s+directors',
        r'appointment\s+of\s+.*officers',
        r'resignation\s+of',
        r'termination\s+of\s+.*officer',
    ]

    # Executive title patterns
    EXECUTIVE_TITLES = {
        'CEO': ['chief executive officer', 'ceo', 'president and ceo'],
        'CFO': ['chief financial officer', 'cfo', 'principal financial officer'],
        'COO': ['chief operating officer', 'coo'],
        'CMO': ['chief medical officer', 'cmo'],
        'CSO': ['chief scientific officer', 'cso', 'chief science officer'],
        'CTO': ['chief technology officer', 'cto'],
        'CCO': ['chief commercial officer', 'chief compliance officer', 'cco'],
        'General Counsel': ['general counsel', 'chief legal officer', 'clo'],
        'Director': ['director', 'board member', 'board of directors'],
        'VP': ['vice president', 'vp', 'svp', 'evp'],
        'President': ['president'],
    }

    # Signal weights based on role and change type
    SIGNAL_WEIGHTS = {
        # Departures (generally negative)
        'CEO_DEPARTURE': -6,
        'CFO_DEPARTURE': -5,
        'CMO_DEPARTURE': -6,  # Critical for biotech
        'CSO_DEPARTURE': -5,  # Critical for biotech
        'COO_DEPARTURE': -4,
        'CCO_DEPARTURE': -3,
        'VP_DEPARTURE': -2,
        'DIRECTOR_DEPARTURE': -2,
        'GENERAL_COUNSEL_DEPARTURE': -3,

        # Appointments (generally positive)
        'CEO_APPOINTMENT': 3,
        'CFO_APPOINTMENT': 2,
        'CMO_APPOINTMENT': 4,  # Positive for biotech pipeline
        'CSO_APPOINTMENT': 3,
        'COO_APPOINTMENT': 2,
        'CCO_APPOINTMENT': 3,  # Could indicate commercial preparation
        'VP_APPOINTMENT': 1,
        'DIRECTOR_APPOINTMENT': 1,

        # Special cases
        'INTERIM_APPOINTMENT': -2,  # Uncertainty signal
        'MASS_EXODUS': -8,  # 3+ departures in 30 days
        'UNEXPECTED_DEPARTURE': -3,  # Additional penalty
        'RETIREMENT': 0,  # Neutral (planned)
    }

    # Departure reason patterns
    DEPARTURE_PATTERNS = {
        'retirement': [r'retire[sd]?', r'retirement'],
        'resignation': [r'resign[sed]', r'resignation', r'stepped down', r'stepping down'],
        'termination': [r'terminat[ed]', r'dismissal', r'fired', r'let go'],
        'health': [r'health reasons', r'medical', r'personal health'],
        'personal': [r'personal reasons', r'family', r'pursue other', r'spend time'],
        'mutual': [r'mutual agreement', r'mutually agreed', r'mutual decision'],
        'disagreement': [r'disagreement', r'difference of opinion', r'strategic differences'],
        'misconduct': [r'misconduct', r'cause', r'violation', r'investigation'],
    }

    def __init__(self, config_path: Optional[str] = None, use_ai: bool = True):
        """
        Initialize the Form 8-K scraper.

        Args:
            config_path: Path to configuration file
            use_ai: Whether to use AI for analysis (requires API key)
        """
        self.config = get_config(config_path)
        self.db = get_database(config_path)
        self.session = self._create_session()
        self.rate_limit = self.config.sec_edgar.get('rate_limit_requests_per_second', 10)
        self.last_request_time = 0
        self.use_ai = use_ai and bool(self.config.anthropic_api_key)

        # CIK mapping
        self._ticker_to_cik: Dict[str, str] = {}

    def _create_session(self) -> requests.Session:
        """Create a requests session with proper SEC headers."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': self.config.sec_user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        })
        return session

    def _rate_limit_wait(self):
        """Ensure we don't exceed SEC's rate limit."""
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Dict = None) -> Optional[requests.Response]:
        """Make a rate-limited request to SEC."""
        self._rate_limit_wait()
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.config.scraping.get('timeout_seconds', 30)
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def load_ticker_cik_mapping(self) -> Dict[str, str]:
        """Load mapping of tickers to CIK numbers from SEC."""
        if self._ticker_to_cik:
            return self._ticker_to_cik

        try:
            url = f"{self.BASE_URL}/files/company_tickers.json"
            response = self._make_request(url)
            if response:
                data = response.json()
                for entry in data.values():
                    ticker = entry.get('ticker', '').upper()
                    cik = str(entry.get('cik_str', '')).zfill(10)
                    if ticker and cik:
                        self._ticker_to_cik[ticker] = cik

                logger.info(f"Loaded {len(self._ticker_to_cik)} ticker-to-CIK mappings")
        except Exception as e:
            logger.error(f"Failed to load ticker-CIK mapping: {e}")

        return self._ticker_to_cik

    def get_cik_for_ticker(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker."""
        if not self._ticker_to_cik:
            self.load_ticker_cik_mapping()
        return self._ticker_to_cik.get(ticker.upper())

    def get_recent_8k_filings(self, cik: str, days: int = 30) -> List[Form8KFiling]:
        """
        Get recent Form 8-K filings for a company.

        Args:
            cik: Company CIK number
            days: Number of days to look back

        Returns:
            List of Form8KFiling objects
        """
        cik_padded = cik.zfill(10)
        url = f"{self.BASE_URL}/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'CIK': cik_padded,
            'type': '8-K',
            'dateb': '',
            'owner': 'include',
            'count': 50,
            'output': 'atom'
        }

        response = self._make_request(url, params)
        if not response:
            return []

        filings = self._parse_atom_feed(response.text, days)
        logger.info(f"Found {len(filings)} 8-K filings for CIK {cik}")
        return filings

    def _parse_atom_feed(self, xml_content: str, days: int) -> List[Form8KFiling]:
        """Parse SEC EDGAR Atom feed for 8-K filings."""
        filings = []
        cutoff_date = datetime.now() - timedelta(days=days)

        try:
            root = ET.fromstring(xml_content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            for entry in root.findall('atom:entry', ns):
                updated = entry.find('atom:updated', ns)
                if updated is not None:
                    filing_date = datetime.fromisoformat(
                        updated.text.replace('Z', '+00:00')
                    ).replace(tzinfo=None)

                    if filing_date < cutoff_date:
                        continue

                    link = entry.find('atom:link', ns)
                    title = entry.find('atom:title', ns)
                    summary = entry.find('atom:summary', ns)

                    if link is not None:
                        href = link.get('href', '')
                        title_text = title.text if title is not None else ''
                        summary_text = summary.text if summary is not None else ''

                        # Extract CIK from URL
                        cik_match = re.search(r'/data/(\d+)/', href)
                        cik = cik_match.group(1) if cik_match else ''

                        # Extract company name
                        company_match = re.search(r'8-K\s*-\s*(.+?)(?:\s*\(|$)', title_text)
                        company_name = company_match.group(1).strip() if company_match else ''

                        # Extract accession number
                        acc_match = re.search(r'(\d+-\d+-\d+)', href)
                        accession = acc_match.group(1) if acc_match else ''

                        # Check for Item 5.02 in summary
                        items = self._extract_items(summary_text)
                        has_502 = any('5.02' in item for item in items)

                        if '/Archives/edgar/data/' in href:
                            filings.append(Form8KFiling(
                                accession_number=accession,
                                company_name=company_name,
                                company_cik=cik,
                                filing_date=filing_date.date(),
                                filing_url=href,
                                items=items,
                                has_item_502=has_502
                            ))

        except ET.ParseError as e:
            logger.error(f"Failed to parse Atom feed: {e}")

        return filings

    def _extract_items(self, summary: str) -> List[str]:
        """Extract item numbers from filing summary."""
        items = []
        if summary:
            # Look for item patterns like "Item 5.02" or "5.02"
            matches = re.findall(r'(?:item\s*)?(\d+\.\d+)', summary.lower())
            items = list(set(matches))
        return items

    def get_8k_document_url(self, filing_page_url: str) -> Optional[str]:
        """Get the URL of the actual 8-K document from the filing page."""
        response = self._make_request(filing_page_url)
        if not response:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for the 8-K document
        for link in soup.find_all('a'):
            href = link.get('href', '')
            text = link.get_text().lower()

            # Primary document is usually labeled "8-K" or ends with certain patterns
            if '8-k' in text or href.endswith('.htm') or href.endswith('.txt'):
                if 'ex' not in href.lower() and 'exhibit' not in text:  # Not an exhibit
                    if not href.startswith('http'):
                        href = f"{self.BASE_URL}{href}"
                    return href

        return None

    def parse_8k_document(self, doc_url: str, ticker: str, filing: Form8KFiling) -> List[ExecutiveChange]:
        """
        Parse Form 8-K document for executive changes.

        Args:
            doc_url: URL of the 8-K document
            ticker: Company ticker
            filing: Form8KFiling metadata

        Returns:
            List of ExecutiveChange objects
        """
        response = self._make_request(doc_url)
        if not response:
            return []

        # Parse HTML/text
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)

        # Check if this contains Item 5.02
        if not self._contains_item_502(text):
            return []

        # Extract Item 5.02 section
        section_text = self._extract_item_502_section(text)
        if not section_text:
            return []

        # Parse executive changes from the section
        changes = self._extract_executive_changes(
            section_text, ticker, filing
        )

        # AI analysis if enabled
        if self.use_ai and changes:
            for change in changes:
                change.ai_analysis = self._analyze_with_ai(section_text, change)

        return changes

    def _contains_item_502(self, text: str) -> bool:
        """Check if text contains Item 5.02."""
        text_lower = text.lower()
        for pattern in self.ITEM_502_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    def _extract_item_502_section(self, text: str) -> str:
        """Extract the Item 5.02 section from the document."""
        text_lower = text.lower()

        # Find start of Item 5.02
        start_patterns = [
            r'item\s*5\.02',
            r'departure\s+of\s+directors\s+or\s+.*officers',
        ]

        start_pos = None
        for pattern in start_patterns:
            match = re.search(pattern, text_lower)
            if match:
                start_pos = match.start()
                break

        if start_pos is None:
            return ''

        # Find end (next item or signature)
        end_patterns = [
            r'item\s*[5-9]\.\d{2}',
            r'item\s*\d+\.',
            r'signature',
            r'pursuant\s+to\s+the\s+requirements',
        ]

        end_pos = len(text)
        remaining_text = text_lower[start_pos + 20:]  # Skip past "Item 5.02"

        for pattern in end_patterns:
            match = re.search(pattern, remaining_text)
            if match:
                end_pos = min(end_pos, start_pos + 20 + match.start())

        section = text[start_pos:end_pos]

        # Limit length
        return section[:10000]

    def _extract_executive_changes(
        self,
        section_text: str,
        ticker: str,
        filing: Form8KFiling
    ) -> List[ExecutiveChange]:
        """Extract executive change information from Item 5.02 section."""
        changes = []
        text_lower = section_text.lower()

        # Determine change type
        is_departure = any(re.search(p, text_lower) for p in [
            r'resign', r'depart', r'terminat', r'retire', r'step.*down', r'left', r'leaving'
        ])
        is_appointment = any(re.search(p, text_lower) for p in [
            r'appoint', r'elect', r'hire', r'nam[ed].*as', r'promot'
        ])

        # Find executive names and titles
        executives = self._find_executives(section_text)

        for exec_name, exec_title in executives:
            # Normalize title
            normalized_title = self._normalize_title(exec_title)

            if is_departure:
                change_type = 'Departure'
                # Check for specific departure types
                if re.search(r'resign', text_lower):
                    change_type = 'Resignation'
                elif re.search(r'terminat|dismiss|fire', text_lower):
                    change_type = 'Termination'
                elif re.search(r'retire', text_lower):
                    change_type = 'Retirement'
            elif is_appointment:
                change_type = 'Appointment'
                if re.search(r'promot', text_lower):
                    change_type = 'Promotion'
            else:
                change_type = 'Unknown'

            # Determine if voluntary
            is_voluntary = self._determine_if_voluntary(text_lower, change_type)

            # Extract reason
            reason = self._extract_reason(text_lower)

            # Find effective date
            effective_date = self._extract_date(section_text)

            # Find successor if this is a departure
            successor = None
            if is_departure:
                successor = self._find_successor(section_text, normalized_title)

            # Calculate signal weight and severity
            signal_weight, severity = self._calculate_signal_weight(
                normalized_title, change_type, is_voluntary, section_text
            )

            changes.append(ExecutiveChange(
                company_ticker=ticker,
                company_cik=filing.company_cik,
                executive_name=exec_name,
                title=normalized_title or exec_title,
                change_type=change_type,
                effective_date=effective_date,
                announcement_date=filing.filing_date,
                reason=reason,
                is_voluntary=is_voluntary,
                successor_name=successor,
                filing_url=filing.filing_url,
                filing_text=section_text[:2000],
                severity_score=severity,
                signal_weight=signal_weight
            ))

        return changes

    def _find_executives(self, text: str) -> List[Tuple[str, str]]:
        """Find executive names and titles in text."""
        executives = []

        # Pattern: "Name, Title" or "Name as Title" or "Title Name"
        patterns = [
            # "John Smith, Chief Executive Officer"
            r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+(?:-[A-Z][a-z]+)?),?\s+(?:the\s+)?(?:Company\'?s?\s+)?([A-Za-z\s]+(?:Officer|Director|President|Counsel|VP|Vice President))',
            # "Chief Executive Officer John Smith"
            r'([A-Za-z\s]+(?:Officer|Director|President|Counsel))\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)',
            # "Mr./Ms. Smith" with title
            r'(?:Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-z]+(?:-[A-Z][a-z]+)?),?\s+(?:the\s+)?(?:Company\'?s?\s+)?([A-Za-z\s]+(?:Officer|Director|President|CEO|CFO|CMO|CSO))',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match[0].strip() if isinstance(match, tuple) else match
                title = match[1].strip() if len(match) > 1 else ''

                # Clean up
                name = re.sub(r'\s+', ' ', name)
                title = re.sub(r'\s+', ' ', title)

                if len(name) > 3 and len(name) < 50:
                    executives.append((name, title))

        # Deduplicate
        seen = set()
        unique = []
        for name, title in executives:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                unique.append((name, title))

        return unique

    def _normalize_title(self, title: str) -> str:
        """Normalize executive title to standard form."""
        title_lower = title.lower()

        for standard_title, patterns in self.EXECUTIVE_TITLES.items():
            for pattern in patterns:
                if pattern in title_lower:
                    return standard_title

        return title

    def _determine_if_voluntary(self, text: str, change_type: str) -> Optional[bool]:
        """Determine if a departure was voluntary."""
        if change_type == 'Retirement':
            return True
        if change_type == 'Termination':
            return False

        # Look for indicators
        voluntary_indicators = [
            'pursue other', 'personal reasons', 'family', 'retire',
            'mutual agreement', 'own decision', 'voluntarily'
        ]
        involuntary_indicators = [
            'for cause', 'terminat', 'dismiss', 'misconduct',
            'let go', 'removed', 'investigation'
        ]

        for indicator in voluntary_indicators:
            if indicator in text:
                return True

        for indicator in involuntary_indicators:
            if indicator in text:
                return False

        return None  # Unknown

    def _extract_reason(self, text: str) -> Optional[str]:
        """Extract the reason for departure/appointment."""
        for reason_type, patterns in self.DEPARTURE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return reason_type

        return None

    def _extract_date(self, text: str) -> Optional[date]:
        """Extract effective date from text."""
        # Common date patterns
        date_patterns = [
            r'effective\s+(?:as\s+of\s+)?(\w+\s+\d{1,2},?\s+\d{4})',
            r'(?:on|as of)\s+(\w+\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(\w+\s+\d{1,2},?\s+\d{4})',
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                try:
                    # Try various formats
                    for fmt in ['%B %d, %Y', '%B %d %Y', '%m/%d/%Y', '%m/%d/%y']:
                        try:
                            return datetime.strptime(date_str, fmt).date()
                        except ValueError:
                            continue
                except Exception:
                    pass

        return None

    def _find_successor(self, text: str, title: str) -> Optional[str]:
        """Find successor name if mentioned."""
        # Look for patterns like "will be succeeded by" or "replacement"
        successor_patterns = [
            r'succeed(?:ed)?\s+by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)',
            r'replac(?:ed|ement)\s+(?:by\s+)?([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)',
            r'appoint(?:ed|ment)\s+(?:of\s+)?([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)\s+as\s+' + re.escape(title.lower()),
        ]

        for pattern in successor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _calculate_signal_weight(
        self,
        title: str,
        change_type: str,
        is_voluntary: Optional[bool],
        text: str
    ) -> Tuple[int, int]:
        """
        Calculate signal weight and severity score.

        Returns:
            Tuple of (signal_weight, severity_score)
        """
        # Base weight from role and change type
        weight_key = f"{title.upper().replace(' ', '_')}_{change_type.upper()}"
        base_weight = self.SIGNAL_WEIGHTS.get(weight_key, 0)

        if base_weight == 0:
            # Try simpler matching
            if 'departure' in change_type.lower() or 'resignation' in change_type.lower():
                if 'CEO' in title:
                    base_weight = self.SIGNAL_WEIGHTS['CEO_DEPARTURE']
                elif 'CFO' in title:
                    base_weight = self.SIGNAL_WEIGHTS['CFO_DEPARTURE']
                elif 'CMO' in title:
                    base_weight = self.SIGNAL_WEIGHTS['CMO_DEPARTURE']
                elif 'CSO' in title:
                    base_weight = self.SIGNAL_WEIGHTS['CSO_DEPARTURE']
                elif 'Director' in title:
                    base_weight = self.SIGNAL_WEIGHTS['DIRECTOR_DEPARTURE']
                elif 'VP' in title:
                    base_weight = self.SIGNAL_WEIGHTS['VP_DEPARTURE']
            elif 'appointment' in change_type.lower():
                if 'CEO' in title:
                    base_weight = self.SIGNAL_WEIGHTS['CEO_APPOINTMENT']
                elif 'CMO' in title:
                    base_weight = self.SIGNAL_WEIGHTS['CMO_APPOINTMENT']
                elif 'CCO' in title:
                    base_weight = self.SIGNAL_WEIGHTS['CCO_APPOINTMENT']

        # Check for interim
        if 'interim' in text.lower() and change_type == 'Appointment':
            base_weight += self.SIGNAL_WEIGHTS['INTERIM_APPOINTMENT']

        # Unexpected departure penalty
        if is_voluntary is False and base_weight < 0:
            base_weight += self.SIGNAL_WEIGHTS['UNEXPECTED_DEPARTURE']

        # Calculate severity (1-10 scale)
        severity = min(10, max(1, abs(base_weight)))

        return base_weight, severity

    def _analyze_with_ai(self, text: str, change: ExecutiveChange) -> Optional[Dict]:
        """Use AI to analyze the executive change."""
        if not self.use_ai:
            return None

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)

            prompt = f"""Analyze this SEC Form 8-K executive change disclosure and provide insights:

Executive: {change.executive_name}
Title: {change.title}
Change Type: {change.change_type}
Company: {change.company_ticker}

Filing Text:
{text[:3000]}

Provide a JSON response with:
1. "sentiment": "bullish", "bearish", or "neutral" for the stock
2. "confidence": 0.0-1.0 confidence in your assessment
3. "key_concerns": list of key concerns (if any)
4. "positive_factors": list of positive factors (if any)
5. "summary": 1-2 sentence summary of the impact

Focus on biotech/pharma context - CMO/CSO departures may indicate pipeline issues."""

            response = client.messages.create(
                model=self.config.ai.get('model', 'claude-sonnet-4-5-20250929'),
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            response_text = response.content[0].text
            # Try to extract JSON from response
            import json
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")

        return None

    def save_executive_changes(self, changes: List[ExecutiveChange]) -> int:
        """Save executive changes to database."""
        if not changes:
            return 0

        saved = 0
        for change in changes:
            data = {
                'company_ticker': change.company_ticker,
                'company_cik': change.company_cik,
                'executive_name': change.executive_name,
                'title': change.title,
                'change_type': change.change_type,
                'effective_date': change.effective_date,
                'announcement_date': change.announcement_date,
                'reason': change.reason,
                'is_voluntary': change.is_voluntary,
                'successor_name': change.successor_name,
                'filing_url': change.filing_url,
                'filing_text': change.filing_text[:2000] if change.filing_text else '',
                'ai_analysis': change.ai_analysis,
                'severity_score': change.severity_score,
                'signal_weight': change.signal_weight,
            }

            try:
                self.db.upsert(
                    'executive_changes',
                    data,
                    conflict_columns=['company_ticker', 'executive_name', 'change_type', 'announcement_date'],
                    update_columns=['title', 'reason', 'ai_analysis', 'severity_score', 'signal_weight']
                )
                saved += 1
            except Exception as e:
                logger.error(f"Failed to save executive change: {e}")

        logger.info(f"Saved {saved} executive changes")
        return saved

    def scrape_company(self, ticker: str, days: int = 30) -> List[ExecutiveChange]:
        """
        Scrape Form 8-K filings for executive changes for a company.

        Args:
            ticker: Company ticker symbol
            days: Number of days to look back

        Returns:
            List of ExecutiveChange objects
        """
        cik = self.get_cik_for_ticker(ticker)
        if not cik:
            logger.warning(f"Could not find CIK for ticker {ticker}")
            return []

        logger.info(f"Scraping 8-K filings for {ticker} (CIK: {cik})")

        all_changes = []
        filings = self.get_recent_8k_filings(cik, days)

        # Filter for Item 5.02
        item_502_filings = [f for f in filings if f.has_item_502 or '5.02' in str(f.items)]

        for filing in item_502_filings:
            doc_url = self.get_8k_document_url(filing.filing_url)
            if doc_url:
                changes = self.parse_8k_document(doc_url, ticker, filing)
                all_changes.extend(changes)

        logger.info(f"Found {len(all_changes)} executive changes for {ticker}")
        return all_changes

    def run(self, tickers: Optional[List[str]] = None, days: int = 30) -> Dict[str, Any]:
        """
        Run the Form 8-K scraper for all watchlist companies.

        Args:
            tickers: Optional list of tickers (defaults to watchlist)
            days: Number of days to look back

        Returns:
            Dictionary with run statistics
        """
        if tickers is None:
            tickers = self.config.watchlist

        run_id = self.db.log_scraper_run('form8k')

        total_changes = 0
        total_saved = 0
        errors = []
        company_changes = {}

        for ticker in tickers:
            try:
                changes = self.scrape_company(ticker, days)
                saved = self.save_executive_changes(changes)

                total_changes += len(changes)
                total_saved += saved

                if changes:
                    company_changes[ticker] = [
                        {
                            'executive': c.executive_name,
                            'title': c.title,
                            'type': c.change_type,
                            'signal_weight': c.signal_weight
                        }
                        for c in changes
                    ]

            except Exception as e:
                errors.append({'ticker': ticker, 'error': str(e)})
                logger.error(f"Failed to scrape {ticker}: {e}")

        self.db.update_scraper_run(
            run_id,
            status='completed' if not errors else 'completed_with_errors',
            records_processed=total_changes,
            records_inserted=total_saved,
            errors_count=len(errors),
            error_details={'errors': errors, 'company_changes': company_changes} if errors else {'company_changes': company_changes}
        )

        return {
            'run_id': run_id,
            'tickers_processed': len(tickers),
            'changes_found': total_changes,
            'changes_saved': total_saved,
            'company_changes': company_changes,
            'errors': errors
        }


if __name__ == '__main__':
    # Test the scraper
    scraper = Form8KScraper(use_ai=False)  # Disable AI for testing

    # Test with a few companies
    result = scraper.run(tickers=['MRNA', 'VRTX', 'BIIB'], days=60)

    print(f"\n8-K Scraper Results:")
    print(f"  Tickers processed: {result['tickers_processed']}")
    print(f"  Changes found: {result['changes_found']}")
    print(f"  Changes saved: {result['changes_saved']}")

    if result['company_changes']:
        print(f"\n  Executive Changes:")
        for ticker, changes in result['company_changes'].items():
            for change in changes:
                print(f"    {ticker}: {change['executive']} ({change['title']}) - {change['type']} (weight: {change['signal_weight']})")
