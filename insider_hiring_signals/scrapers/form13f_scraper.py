"""
SEC Form 13F Scraper for Institutional Holdings Data
Tracks positions of top biotech-focused institutional investors
"""

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from utils.config import get_config
from utils.database import get_database
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class InstitutionalHolding:
    """Represents an institutional position from 13F filing."""
    fund_name: str
    fund_cik: str
    company_name: str
    company_ticker: str
    cusip: str
    quarter_end: date
    shares: int
    market_value: int  # In thousands
    shares_prev_quarter: Optional[int]
    pct_change_shares: Optional[float]
    is_new_position: bool
    is_exit: bool
    put_call: Optional[str]
    investment_discretion: str
    filing_date: date
    filing_url: str
    signal_weight: int = 0


class Form13FScraper:
    """
    Scrapes SEC 13F filings for institutional holdings.

    13F filings are required quarterly for institutional investment managers
    with over $100M in assets. They reveal positions 45 days after quarter end.
    """

    BASE_URL = "https://www.sec.gov"
    CUSIP_TO_TICKER_CACHE: Dict[str, str] = {}

    # Signal weights for institutional activity
    SIGNAL_WEIGHTS = {
        'NEW_POSITION': 5,
        'INCREASE_GT_50': 4,
        'INCREASE_GT_25': 2,
        'DECREASE_GT_50': -4,
        'DECREASE_GT_25': -2,
        'EXIT_POSITION': -5,
        'MULTIPLE_FUNDS_INITIATE': 6,
        'MULTIPLE_FUNDS_EXIT': -6,
    }

    # Top biotech funds to track (from config)
    TOP_BIOTECH_FUNDS = [
        {'name': 'Baker Bros Advisors LP', 'cik': '0001537554'},
        {'name': 'RA Capital Management LP', 'cik': '0001598627'},
        {'name': 'Perceptive Advisors LLC', 'cik': '0001428987'},
        {'name': 'Boxer Capital LLC', 'cik': '0001510494'},
        {'name': 'OrbiMed Advisors LLC', 'cik': '0001187508'},
        {'name': 'Farallon Capital Management LLC', 'cik': '0001015780'},
        {'name': 'Viking Global Investors LP', 'cik': '0001103804'},
        {'name': 'Partner Fund Management LP', 'cik': '0001484967'},
        {'name': 'Deerfield Management Company', 'cik': '0001502183'},
        {'name': 'Tang Capital Management LLC', 'cik': '0001569691'},
    ]

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.db = get_database(config_path)
        self.session = self._create_session()
        self.rate_limit = self.config.sec_edgar.get('rate_limit_requests_per_second', 10)
        self.last_request_time = 0
        self._watchlist_cusips: Dict[str, str] = {}  # CUSIP -> Ticker

    def _create_session(self) -> requests.Session:
        """Create a requests session with proper headers."""
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

    def _make_request(self, url: str, params: Dict = None) -> requests.Response:
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
            raise

    def load_watchlist_cusips(self):
        """Load CUSIP mappings for watchlist companies."""
        if self._watchlist_cusips:
            return

        # Try to load from SEC's company tickers file
        try:
            url = f"{self.BASE_URL}/files/company_tickers.json"
            response = self._make_request(url)
            data = response.json()

            # Build ticker -> CIK mapping first
            ticker_to_cik = {}
            for entry in data.values():
                ticker = entry.get('ticker', '').upper()
                cik = str(entry.get('cik_str', '')).zfill(10)
                if ticker in self.config.watchlist:
                    ticker_to_cik[ticker] = cik

            logger.info(f"Loaded {len(ticker_to_cik)} watchlist company mappings")

        except Exception as e:
            logger.error(f"Failed to load company data: {e}")

    def get_cusip_ticker_mapping(self, cusip: str) -> Optional[str]:
        """Get ticker for a CUSIP (using cache or lookup)."""
        if cusip in self.CUSIP_TO_TICKER_CACHE:
            return self.CUSIP_TO_TICKER_CACHE[cusip]

        # Check database first
        result = self.db.execute_one(
            "SELECT ticker FROM companies WHERE cusip = %s",
            (cusip,)
        )
        if result:
            self.CUSIP_TO_TICKER_CACHE[cusip] = result['ticker']
            return result['ticker']

        return None

    def get_recent_13f_filings(self, fund_cik: str, quarters: int = 4) -> List[Dict]:
        """
        Get recent 13F filings for an institutional investor.

        Args:
            fund_cik: Fund's CIK number
            quarters: Number of quarters to look back

        Returns:
            List of filing metadata dictionaries
        """
        cik_padded = fund_cik.zfill(10)
        url = f"{self.BASE_URL}/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'CIK': cik_padded,
            'type': '13F-HR',
            'dateb': '',
            'owner': 'include',
            'count': quarters * 2,  # Get extra in case of amendments
            'output': 'atom'
        }

        try:
            response = self._make_request(url, params)
            filings = self._parse_atom_feed(response.text)
            logger.info(f"Found {len(filings)} 13F filings for CIK {fund_cik}")
            return filings[:quarters]  # Return most recent quarters
        except Exception as e:
            logger.error(f"Failed to get 13F filings for CIK {fund_cik}: {e}")
            return []

    def _parse_atom_feed(self, xml_content: str) -> List[Dict]:
        """Parse SEC EDGAR Atom feed for 13F filings."""
        filings = []

        try:
            root = ET.fromstring(xml_content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            for entry in root.findall('atom:entry', ns):
                updated = entry.find('atom:updated', ns)
                filing_date = None
                if updated is not None:
                    filing_date = datetime.fromisoformat(
                        updated.text.replace('Z', '+00:00')
                    ).replace(tzinfo=None).date()

                link = entry.find('atom:link', ns)
                title = entry.find('atom:title', ns)

                if link is not None:
                    href = link.get('href', '')
                    title_text = title.text if title is not None else ''

                    # Skip amendments for now (can add later)
                    if '/A' in title_text:
                        continue

                    if '/Archives/edgar/data/' in href:
                        filings.append({
                            'filing_date': filing_date,
                            'url': href,
                            'title': title_text
                        })

        except ET.ParseError as e:
            logger.error(f"Failed to parse Atom feed: {e}")

        return filings

    def get_13f_info_table_url(self, filing_page_url: str) -> Optional[str]:
        """Extract the information table XML URL from a 13F filing page."""
        try:
            response = self._make_request(filing_page_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Look for the infotable.xml file
            for link in soup.find_all('a'):
                href = link.get('href', '')
                text = link.get_text().lower()

                # 13F information table is usually named infotable.xml
                if 'infotable' in href.lower() or 'infotable' in text:
                    if href.endswith('.xml'):
                        if not href.startswith('http'):
                            href = f"{self.BASE_URL}{href}"
                        return href

            # Fallback: look for any XML that might be the table
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href.endswith('.xml') and 'primary' not in href.lower():
                    if not href.startswith('http'):
                        href = f"{self.BASE_URL}{href}"
                    return href

        except Exception as e:
            logger.error(f"Failed to get 13F info table URL from {filing_page_url}: {e}")

        return None

    def parse_13f_xml(self, xml_url: str, fund_name: str, fund_cik: str,
                      filing_date: date) -> List[InstitutionalHolding]:
        """
        Parse 13F information table XML.

        Args:
            xml_url: URL to the 13F information table XML
            fund_name: Name of the institutional investor
            fund_cik: CIK of the fund
            filing_date: Date the filing was submitted

        Returns:
            List of InstitutionalHolding objects
        """
        holdings = []

        try:
            response = self._make_request(xml_url)
            content = response.content

            # Parse XML (handle namespace)
            root = ET.fromstring(content)

            # Determine quarter end from filing date
            quarter_end = self._get_quarter_end(filing_date)

            # Find all info table entries
            # Namespace handling for 13F XML
            ns_map = {
                'ns1': 'http://www.sec.gov/edgar/document/thirteenf/informationtable',
                '': 'http://www.sec.gov/edgar/document/thirteenf/informationtable'
            }

            # Try different XPath patterns for the info table
            info_tables = (
                root.findall('.//ns1:infoTable', ns_map) or
                root.findall('.//{http://www.sec.gov/edgar/document/thirteenf/informationtable}infoTable') or
                root.findall('.//infoTable')
            )

            for entry in info_tables:
                holding = self._parse_info_table_entry(
                    entry, fund_name, fund_cik, quarter_end, filing_date, xml_url, ns_map
                )
                if holding:
                    # Only include holdings for watchlist companies
                    if holding.company_ticker in self.config.watchlist:
                        holdings.append(holding)

            logger.debug(f"Parsed {len(holdings)} holdings from {xml_url}")

        except ET.ParseError as e:
            logger.error(f"Failed to parse 13F XML at {xml_url}: {e}")
        except Exception as e:
            logger.error(f"Error processing 13F at {xml_url}: {e}")

        return holdings

    def _parse_info_table_entry(self, entry: ET.Element, fund_name: str,
                                 fund_cik: str, quarter_end: date,
                                 filing_date: date, filing_url: str,
                                 ns_map: Dict) -> Optional[InstitutionalHolding]:
        """Parse a single info table entry from 13F."""
        try:
            # Helper to get text from element
            def get_text(elem, tag):
                for ns_prefix in ['ns1:', '', '{http://www.sec.gov/edgar/document/thirteenf/informationtable}']:
                    child = elem.find(f'{ns_prefix}{tag}', ns_map) if ns_prefix.startswith('ns') else elem.find(f'{ns_prefix}{tag}')
                    if child is not None and child.text:
                        return child.text.strip()
                return None

            # Extract fields
            company_name = get_text(entry, 'nameOfIssuer') or ''
            cusip = get_text(entry, 'cusip') or ''
            value = get_text(entry, 'value') or '0'
            shares = get_text(entry, 'shrsOrPrnAmt/sshPrnamt') or get_text(entry, 'sshPrnamt') or '0'
            put_call = get_text(entry, 'putCall')
            investment_discretion = get_text(entry, 'investmentDiscretion') or 'SOLE'

            # Try to get ticker from CUSIP
            ticker = self.get_cusip_ticker_mapping(cusip)
            if not ticker:
                # Try to match by company name
                ticker = self._match_ticker_by_name(company_name)

            if not ticker:
                return None  # Skip if we can't identify the company

            return InstitutionalHolding(
                fund_name=fund_name,
                fund_cik=fund_cik,
                company_name=company_name,
                company_ticker=ticker,
                cusip=cusip,
                quarter_end=quarter_end,
                shares=int(shares.replace(',', '')),
                market_value=int(value.replace(',', '')),  # Value is in thousands
                shares_prev_quarter=None,  # Will be filled in later
                pct_change_shares=None,
                is_new_position=False,
                is_exit=False,
                put_call=put_call,
                investment_discretion=investment_discretion,
                filing_date=filing_date,
                filing_url=filing_url,
                signal_weight=0
            )

        except Exception as e:
            logger.error(f"Failed to parse info table entry: {e}")
            return None

    def _match_ticker_by_name(self, company_name: str) -> Optional[str]:
        """Try to match a ticker based on company name."""
        company_name_lower = company_name.lower()

        # Simple matching for common biotech companies
        name_to_ticker = {
            'moderna': 'MRNA',
            'biontech': 'BNTX',
            'vertex': 'VRTX',
            'regeneron': 'REGN',
            'biogen': 'BIIB',
            'alnylam': 'ALNY',
            'biomarin': 'BMRN',
            'incyte': 'INCY',
            'ionis': 'IONS',
            'neurocrine': 'NBIX',
            'crispr': 'CRSP',
            'beam': 'BEAM',
            'editas': 'EDIT',
            'intellia': 'NTLA',
            'fate': 'FATE',
            'arcus': 'ARCUS',
            'sage': 'SAGE',
            'blueprint': 'BPMC',
            'seagen': 'SGEN',
        }

        for name_part, ticker in name_to_ticker.items():
            if name_part in company_name_lower:
                return ticker

        return None

    def _get_quarter_end(self, filing_date: date) -> date:
        """Determine quarter end date from filing date."""
        # 13F is filed within 45 days of quarter end
        # So we go back ~45 days and find the quarter end

        target_date = filing_date - timedelta(days=45)
        quarter_ends = [
            date(target_date.year, 3, 31),
            date(target_date.year, 6, 30),
            date(target_date.year, 9, 30),
            date(target_date.year, 12, 31),
            date(target_date.year - 1, 12, 31),
        ]

        # Find the closest quarter end before the target date
        valid_quarters = [q for q in quarter_ends if q <= filing_date]
        if valid_quarters:
            return max(valid_quarters)

        return date(target_date.year, 12, 31)

    def calculate_position_changes(self, fund_cik: str, current_quarter: date,
                                   holdings: List[InstitutionalHolding]) -> List[InstitutionalHolding]:
        """
        Calculate position changes vs previous quarter.

        Args:
            fund_cik: Fund's CIK number
            current_quarter: Current quarter end date
            holdings: List of current holdings

        Returns:
            Holdings with change calculations populated
        """
        # Get previous quarter holdings from database
        prev_quarter = self._get_previous_quarter(current_quarter)

        prev_holdings = self.db.execute("""
            SELECT company_ticker, shares
            FROM institutional_holdings
            WHERE fund_cik = %s AND quarter_end = %s
        """, (fund_cik, prev_quarter))

        prev_shares_map = {h['company_ticker']: h['shares'] for h in prev_holdings}

        # Current tickers
        current_tickers = {h.company_ticker for h in holdings}

        # Calculate changes
        for holding in holdings:
            ticker = holding.company_ticker
            prev_shares = prev_shares_map.get(ticker)

            if prev_shares is None:
                # New position
                holding.is_new_position = True
                holding.shares_prev_quarter = 0
                holding.pct_change_shares = 100.0  # New positions
                holding.signal_weight = self.SIGNAL_WEIGHTS['NEW_POSITION']
            else:
                holding.shares_prev_quarter = prev_shares
                if prev_shares > 0:
                    holding.pct_change_shares = ((holding.shares - prev_shares) / prev_shares) * 100

                    # Assign signal weight based on change magnitude
                    if holding.pct_change_shares > 50:
                        holding.signal_weight = self.SIGNAL_WEIGHTS['INCREASE_GT_50']
                    elif holding.pct_change_shares > 25:
                        holding.signal_weight = self.SIGNAL_WEIGHTS['INCREASE_GT_25']
                    elif holding.pct_change_shares < -50:
                        holding.signal_weight = self.SIGNAL_WEIGHTS['DECREASE_GT_50']
                    elif holding.pct_change_shares < -25:
                        holding.signal_weight = self.SIGNAL_WEIGHTS['DECREASE_GT_25']

        # Check for exits (positions in prev quarter but not current)
        for ticker, prev_shares in prev_shares_map.items():
            if ticker not in current_tickers and ticker in self.config.watchlist:
                # This is an exit
                exit_holding = InstitutionalHolding(
                    fund_name=holdings[0].fund_name if holdings else '',
                    fund_cik=fund_cik,
                    company_name='',
                    company_ticker=ticker,
                    cusip='',
                    quarter_end=current_quarter,
                    shares=0,
                    market_value=0,
                    shares_prev_quarter=prev_shares,
                    pct_change_shares=-100.0,
                    is_new_position=False,
                    is_exit=True,
                    put_call=None,
                    investment_discretion='',
                    filing_date=holdings[0].filing_date if holdings else current_quarter,
                    filing_url='',
                    signal_weight=self.SIGNAL_WEIGHTS['EXIT_POSITION']
                )
                holdings.append(exit_holding)

        return holdings

    def _get_previous_quarter(self, quarter_end: date) -> date:
        """Get the previous quarter end date."""
        month = quarter_end.month
        year = quarter_end.year

        if month == 3:
            return date(year - 1, 12, 31)
        elif month == 6:
            return date(year, 3, 31)
        elif month == 9:
            return date(year, 6, 30)
        else:  # December
            return date(year, 9, 30)

    def scrape_fund(self, fund_name: str, fund_cik: str,
                    quarters: int = 2) -> List[InstitutionalHolding]:
        """
        Scrape 13F filings for a specific fund.

        Args:
            fund_name: Name of the fund
            fund_cik: CIK number of the fund
            quarters: Number of quarters to scrape

        Returns:
            List of InstitutionalHolding objects
        """
        logger.info(f"Scraping 13F filings for {fund_name} (CIK: {fund_cik})")

        all_holdings = []
        filings = self.get_recent_13f_filings(fund_cik, quarters)

        for filing in filings:
            info_table_url = self.get_13f_info_table_url(filing['url'])
            if info_table_url:
                holdings = self.parse_13f_xml(
                    info_table_url, fund_name, fund_cik, filing['filing_date']
                )

                if holdings:
                    # Calculate position changes
                    quarter_end = holdings[0].quarter_end
                    holdings = self.calculate_position_changes(fund_cik, quarter_end, holdings)

                all_holdings.extend(holdings)

        logger.info(f"Found {len(all_holdings)} holdings for {fund_name}")
        return all_holdings

    def save_holdings(self, holdings: List[InstitutionalHolding]) -> Tuple[int, int]:
        """
        Save holdings to the database.

        Returns:
            Tuple of (inserted_count, updated_count)
        """
        if not holdings:
            return 0, 0

        inserted = 0
        updated = 0

        for holding in holdings:
            data = {
                'fund_name': holding.fund_name,
                'fund_cik': holding.fund_cik,
                'company_ticker': holding.company_ticker,
                'company_name': holding.company_name,
                'cusip': holding.cusip,
                'quarter_end': holding.quarter_end,
                'shares': holding.shares,
                'market_value': holding.market_value,
                'shares_prev_quarter': holding.shares_prev_quarter,
                'pct_change_shares': holding.pct_change_shares,
                'is_new_position': holding.is_new_position,
                'is_exit': holding.is_exit,
                'put_call': holding.put_call,
                'investment_discretion': holding.investment_discretion,
                'filing_date': holding.filing_date,
                'filing_url': holding.filing_url,
                'signal_weight': holding.signal_weight,
            }

            try:
                self.db.upsert(
                    'institutional_holdings',
                    data,
                    conflict_columns=['fund_cik', 'company_ticker', 'quarter_end'],
                    update_columns=[
                        'shares', 'market_value', 'shares_prev_quarter',
                        'pct_change_shares', 'is_new_position', 'is_exit',
                        'signal_weight', 'filing_url'
                    ]
                )
                inserted += 1
            except Exception as e:
                logger.error(f"Failed to save holding: {e}")

        logger.info(f"Saved {inserted} holdings")
        return inserted, updated

    def run(self, funds: Optional[List[Dict]] = None, quarters: int = 2) -> Dict[str, Any]:
        """
        Run the 13F scraper for all tracked funds.

        Args:
            funds: Optional list of fund dicts with 'name' and 'cik'
            quarters: Number of quarters to look back

        Returns:
            Dictionary with run statistics
        """
        if funds is None:
            funds = self.config.institutional_investors
            if not funds:
                funds = self.TOP_BIOTECH_FUNDS

        run_id = self.db.log_scraper_run('13f')

        total_holdings = 0
        total_inserted = 0
        errors = []

        for fund in funds:
            try:
                holdings = self.scrape_fund(
                    fund['name'], fund['cik'], quarters
                )
                inserted, _ = self.save_holdings(holdings)
                total_holdings += len(holdings)
                total_inserted += inserted
            except Exception as e:
                errors.append({'fund': fund['name'], 'error': str(e)})
                logger.error(f"Failed to scrape {fund['name']}: {e}")

        self.db.update_scraper_run(
            run_id,
            status='completed' if not errors else 'completed_with_errors',
            records_processed=total_holdings,
            records_inserted=total_inserted,
            errors_count=len(errors),
            error_details={'errors': errors} if errors else None
        )

        return {
            'run_id': run_id,
            'funds_processed': len(funds),
            'holdings_found': total_holdings,
            'holdings_inserted': total_inserted,
            'errors': errors
        }


if __name__ == '__main__':
    # Test the scraper
    scraper = Form13FScraper()

    # Test with one fund
    result = scraper.run(
        funds=[{'name': 'Baker Bros Advisors LP', 'cik': '0001537554'}],
        quarters=1
    )
    print(f"Scraper result: {result}")
