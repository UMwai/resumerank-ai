"""
SEC Form 4 Scraper for Insider Trading Data
Fetches and parses insider transactions from SEC EDGAR
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
class InsiderTransaction:
    """Represents a single insider transaction from Form 4."""
    company_ticker: str
    company_cik: str
    insider_name: str
    insider_cik: str
    insider_title: str
    is_director: bool
    is_officer: bool
    is_ten_percent_owner: bool
    transaction_date: date
    transaction_type: str
    transaction_code: str
    shares: int
    price_per_share: float
    transaction_value: float
    shares_owned_after: int
    ownership_nature: str
    is_10b5_1_plan: bool
    footnotes: str
    filing_date: date
    filing_url: str
    signal_weight: int = 0
    raw_data: Dict = None


class Form4Scraper:
    """
    Scrapes SEC Form 4 filings for insider trading information.

    The SEC requires a User-Agent header and limits requests to 10 per second.
    This scraper respects those limits and implements proper error handling.
    """

    BASE_URL = "https://www.sec.gov"
    EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    # Transaction codes for SEC Form 4
    TRANSACTION_CODES = {
        'P': 'Purchase',
        'S': 'Sale',
        'A': 'Award',
        'D': 'Disposition',
        'F': 'Tax Withholding',
        'I': 'Discretionary',
        'M': 'Option Exercise',
        'C': 'Conversion',
        'E': 'Expiration',
        'G': 'Gift',
        'H': 'Gift Expired',
        'J': 'Other',
        'K': 'Equity Swap',
        'L': 'Small Acquisition',
        'U': 'Disposition to Trust',
        'W': 'Will/Inheritance',
        'Z': 'Trust'
    }

    # Signal weights based on insider role and transaction type
    SIGNAL_WEIGHTS = {
        'CEO_BUY': 5,
        'CFO_BUY': 4,
        'CMO_BUY': 5,
        'DIRECTOR_BUY': 3,
        'OFFICER_BUY': 3,
        'CEO_SELL': -4,
        'CFO_SELL': -5,
        'CMO_SELL': -5,
        'MULTIPLE_INSIDER_BUY': 6,
        'MULTIPLE_INSIDER_SELL': -6,
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.db = get_database(config_path)
        self.session = self._create_session()
        self.rate_limit = self.config.sec_edgar.get('rate_limit_requests_per_second', 10)
        self.last_request_time = 0
        self._ticker_to_cik: Dict[str, str] = {}

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

    def load_ticker_cik_mapping(self) -> Dict[str, str]:
        """Load mapping of tickers to CIK numbers from SEC."""
        if self._ticker_to_cik:
            return self._ticker_to_cik

        try:
            response = self._make_request(self.COMPANY_TICKERS_URL)
            data = response.json()

            for entry in data.values():
                ticker = entry.get('ticker', '').upper()
                cik = str(entry.get('cik_str', '')).zfill(10)
                if ticker and cik:
                    self._ticker_to_cik[ticker] = cik

            logger.info(f"Loaded {len(self._ticker_to_cik)} ticker-to-CIK mappings")
            return self._ticker_to_cik
        except Exception as e:
            logger.error(f"Failed to load ticker-CIK mapping: {e}")
            return {}

    def get_cik_for_ticker(self, ticker: str) -> Optional[str]:
        """Get CIK number for a given ticker."""
        if not self._ticker_to_cik:
            self.load_ticker_cik_mapping()
        return self._ticker_to_cik.get(ticker.upper())

    def get_recent_form4_filings(self, cik: str, days: int = 30) -> List[Dict]:
        """
        Get recent Form 4 filings for a company.

        Args:
            cik: Company CIK number
            days: Number of days to look back

        Returns:
            List of filing metadata dictionaries
        """
        # Use SEC EDGAR company filings endpoint
        cik_padded = cik.zfill(10)
        url = f"{self.BASE_URL}/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'CIK': cik_padded,
            'type': '4',
            'dateb': '',
            'owner': 'include',
            'count': 100,
            'output': 'atom'
        }

        try:
            response = self._make_request(url, params)
            filings = self._parse_atom_feed(response.text, days)
            logger.info(f"Found {len(filings)} Form 4 filings for CIK {cik}")
            return filings
        except Exception as e:
            logger.error(f"Failed to get Form 4 filings for CIK {cik}: {e}")
            return []

    def _parse_atom_feed(self, xml_content: str, days: int) -> List[Dict]:
        """Parse SEC EDGAR Atom feed for Form 4 filings."""
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

                if link is not None:
                    href = link.get('href', '')
                    # Extract the filing URL
                    if '/Archives/edgar/data/' in href:
                        filings.append({
                            'filing_date': filing_date.date(),
                            'url': href,
                            'title': title.text if title is not None else ''
                        })
        except ET.ParseError as e:
            logger.error(f"Failed to parse Atom feed: {e}")

        return filings

    def get_form4_xml_url(self, filing_page_url: str) -> Optional[str]:
        """Extract the XML file URL from a Form 4 filing page."""
        try:
            response = self._make_request(filing_page_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Look for the XML document link
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href.endswith('.xml') and 'primary_doc' not in href.lower():
                    # Prefer non-primary XML (the actual Form 4 data)
                    if not href.startswith('http'):
                        href = f"{self.BASE_URL}{href}"
                    return href

            # Fallback: look for any XML
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href.endswith('.xml'):
                    if not href.startswith('http'):
                        href = f"{self.BASE_URL}{href}"
                    return href

        except Exception as e:
            logger.error(f"Failed to get Form 4 XML URL from {filing_page_url}: {e}")

        return None

    def parse_form4_xml(self, xml_url: str, company_ticker: str,
                        filing_date: date) -> List[InsiderTransaction]:
        """
        Parse Form 4 XML filing and extract transactions.

        Args:
            xml_url: URL to the Form 4 XML file
            company_ticker: Company ticker symbol
            filing_date: Date the filing was submitted

        Returns:
            List of InsiderTransaction objects
        """
        transactions = []

        try:
            response = self._make_request(xml_url)
            root = ET.fromstring(response.content)

            # Extract reporting owner information
            owner_info = self._extract_owner_info(root)
            issuer_info = self._extract_issuer_info(root)

            # Parse non-derivative transactions
            for trans_elem in root.findall('.//nonDerivativeTransaction'):
                trans = self._parse_transaction(
                    trans_elem, owner_info, issuer_info,
                    company_ticker, filing_date, xml_url
                )
                if trans:
                    transactions.append(trans)

            # Parse derivative transactions
            for trans_elem in root.findall('.//derivativeTransaction'):
                trans = self._parse_derivative_transaction(
                    trans_elem, owner_info, issuer_info,
                    company_ticker, filing_date, xml_url
                )
                if trans:
                    transactions.append(trans)

            logger.debug(f"Parsed {len(transactions)} transactions from {xml_url}")

        except ET.ParseError as e:
            logger.error(f"Failed to parse Form 4 XML at {xml_url}: {e}")
        except Exception as e:
            logger.error(f"Error processing Form 4 at {xml_url}: {e}")

        return transactions

    def _extract_owner_info(self, root: ET.Element) -> Dict[str, Any]:
        """Extract reporting owner information from Form 4 XML."""
        info = {
            'name': '',
            'cik': '',
            'is_director': False,
            'is_officer': False,
            'is_ten_percent_owner': False,
            'title': ''
        }

        owner = root.find('.//reportingOwner')
        if owner is not None:
            owner_id = owner.find('reportingOwnerId')
            if owner_id is not None:
                name_elem = owner_id.find('rptOwnerName')
                cik_elem = owner_id.find('rptOwnerCik')
                info['name'] = name_elem.text if name_elem is not None else ''
                info['cik'] = cik_elem.text if cik_elem is not None else ''

            relationship = owner.find('reportingOwnerRelationship')
            if relationship is not None:
                is_director = relationship.find('isDirector')
                is_officer = relationship.find('isOfficer')
                is_ten_pct = relationship.find('isTenPercentOwner')
                title = relationship.find('officerTitle')

                info['is_director'] = is_director is not None and is_director.text == '1'
                info['is_officer'] = is_officer is not None and is_officer.text == '1'
                info['is_ten_percent_owner'] = is_ten_pct is not None and is_ten_pct.text == '1'
                info['title'] = title.text if title is not None else ''

        return info

    def _extract_issuer_info(self, root: ET.Element) -> Dict[str, str]:
        """Extract issuer information from Form 4 XML."""
        info = {'cik': '', 'name': '', 'ticker': ''}

        issuer = root.find('.//issuer')
        if issuer is not None:
            cik = issuer.find('issuerCik')
            name = issuer.find('issuerName')
            ticker = issuer.find('issuerTradingSymbol')

            info['cik'] = cik.text if cik is not None else ''
            info['name'] = name.text if name is not None else ''
            info['ticker'] = ticker.text if ticker is not None else ''

        return info

    def _parse_transaction(self, trans_elem: ET.Element, owner_info: Dict,
                           issuer_info: Dict, company_ticker: str,
                           filing_date: date, filing_url: str) -> Optional[InsiderTransaction]:
        """Parse a non-derivative transaction element."""
        try:
            # Transaction date
            date_elem = trans_elem.find('.//transactionDate/value')
            trans_date = date.fromisoformat(date_elem.text) if date_elem is not None else filing_date

            # Transaction coding
            coding = trans_elem.find('.//transactionCoding')
            trans_code = ''
            is_10b5_1 = False

            if coding is not None:
                code_elem = coding.find('transactionCode')
                trans_code = code_elem.text if code_elem is not None else ''

                # Check for 10b5-1 plan indicator
                form_type = coding.find('transactionFormType')
                equity_swap = coding.find('equitySwapInvolved')
                # Look for footnotes indicating 10b5-1
                footnote_ids = trans_elem.findall('.//footnoteId')
                is_10b5_1 = any(self._check_10b5_1_footnote(trans_elem, f.get('id', ''))
                                for f in footnote_ids)

            # Transaction amounts
            amounts = trans_elem.find('.//transactionAmounts')
            shares = 0
            price = 0.0

            if amounts is not None:
                shares_elem = amounts.find('transactionShares/value')
                price_elem = amounts.find('transactionPricePerShare/value')

                shares = int(float(shares_elem.text)) if shares_elem is not None and shares_elem.text else 0
                price = float(price_elem.text) if price_elem is not None and price_elem.text else 0.0

            # Post-transaction ownership
            post_ownership = trans_elem.find('.//postTransactionAmounts/sharesOwnedFollowingTransaction/value')
            shares_after = int(float(post_ownership.text)) if post_ownership is not None and post_ownership.text else 0

            # Ownership nature
            ownership_nature_elem = trans_elem.find('.//ownershipNature/directOrIndirectOwnership/value')
            ownership_nature = ownership_nature_elem.text if ownership_nature_elem is not None else 'D'
            ownership_nature = 'Direct' if ownership_nature == 'D' else 'Indirect'

            # Determine transaction type
            trans_type = self.TRANSACTION_CODES.get(trans_code, 'Other')

            # Calculate transaction value
            trans_value = shares * price

            # Calculate signal weight
            signal_weight = self._calculate_signal_weight(
                owner_info, trans_type, trans_value, is_10b5_1
            )

            # Extract footnotes
            footnotes = self._extract_footnotes(trans_elem)

            return InsiderTransaction(
                company_ticker=company_ticker,
                company_cik=issuer_info.get('cik', ''),
                insider_name=owner_info.get('name', ''),
                insider_cik=owner_info.get('cik', ''),
                insider_title=owner_info.get('title', ''),
                is_director=owner_info.get('is_director', False),
                is_officer=owner_info.get('is_officer', False),
                is_ten_percent_owner=owner_info.get('is_ten_percent_owner', False),
                transaction_date=trans_date,
                transaction_type=trans_type,
                transaction_code=trans_code,
                shares=shares,
                price_per_share=price,
                transaction_value=trans_value,
                shares_owned_after=shares_after,
                ownership_nature=ownership_nature,
                is_10b5_1_plan=is_10b5_1,
                footnotes=footnotes,
                filing_date=filing_date,
                filing_url=filing_url,
                signal_weight=signal_weight
            )

        except Exception as e:
            logger.error(f"Failed to parse transaction: {e}")
            return None

    def _parse_derivative_transaction(self, trans_elem: ET.Element, owner_info: Dict,
                                       issuer_info: Dict, company_ticker: str,
                                       filing_date: date, filing_url: str) -> Optional[InsiderTransaction]:
        """Parse a derivative transaction element (options, warrants, etc.)."""
        try:
            # Transaction date
            date_elem = trans_elem.find('.//transactionDate/value')
            trans_date = date.fromisoformat(date_elem.text) if date_elem is not None else filing_date

            # Transaction coding
            coding = trans_elem.find('.//transactionCoding')
            trans_code = ''
            is_10b5_1 = False

            if coding is not None:
                code_elem = coding.find('transactionCode')
                trans_code = code_elem.text if code_elem is not None else ''

            # For derivatives, we typically look at conversion/exercise
            amounts = trans_elem.find('.//transactionAmounts')
            shares = 0

            if amounts is not None:
                shares_elem = amounts.find('transactionShares/value')
                shares = int(float(shares_elem.text)) if shares_elem is not None and shares_elem.text else 0

            # Get exercise price
            exercise_price_elem = trans_elem.find('.//conversionOrExercisePrice/value')
            price = float(exercise_price_elem.text) if exercise_price_elem is not None and exercise_price_elem.text else 0.0

            trans_type = self.TRANSACTION_CODES.get(trans_code, 'Option Exercise')
            trans_value = shares * price

            # Lower signal weight for derivative transactions (usually options)
            signal_weight = 0  # Options are generally less informative

            return InsiderTransaction(
                company_ticker=company_ticker,
                company_cik=issuer_info.get('cik', ''),
                insider_name=owner_info.get('name', ''),
                insider_cik=owner_info.get('cik', ''),
                insider_title=owner_info.get('title', ''),
                is_director=owner_info.get('is_director', False),
                is_officer=owner_info.get('is_officer', False),
                is_ten_percent_owner=owner_info.get('is_ten_percent_owner', False),
                transaction_date=trans_date,
                transaction_type=trans_type,
                transaction_code=trans_code,
                shares=shares,
                price_per_share=price,
                transaction_value=trans_value,
                shares_owned_after=0,
                ownership_nature='Direct',
                is_10b5_1_plan=is_10b5_1,
                footnotes='',
                filing_date=filing_date,
                filing_url=filing_url,
                signal_weight=signal_weight
            )

        except Exception as e:
            logger.error(f"Failed to parse derivative transaction: {e}")
            return None

    def _check_10b5_1_footnote(self, trans_elem: ET.Element, footnote_id: str) -> bool:
        """Check if a footnote indicates a 10b5-1 plan."""
        # Look for footnotes in the root document
        root = trans_elem
        while root.getparent() is not None:
            root = root.getparent()

        footnotes = root.findall(f".//footnote[@id='{footnote_id}']")
        for fn in footnotes:
            text = fn.text or ''
            if '10b5-1' in text.lower() or 'rule 10b5-1' in text.lower():
                return True
        return False

    def _extract_footnotes(self, trans_elem: ET.Element) -> str:
        """Extract footnote text related to a transaction."""
        footnotes = []
        footnote_ids = trans_elem.findall('.//footnoteId')

        for fn_id in footnote_ids:
            fn_ref = fn_id.get('id', '')
            # Note: Full footnote extraction would require access to root
            footnotes.append(f"[{fn_ref}]")

        return ' '.join(footnotes)

    def _calculate_signal_weight(self, owner_info: Dict, trans_type: str,
                                  trans_value: float, is_10b5_1: bool) -> int:
        """
        Calculate signal weight based on insider role and transaction.

        Key rules:
        - Open market purchases are highly informative
        - 10b5-1 plan transactions are less informative (pre-scheduled)
        - C-suite transactions carry more weight
        - Minimum transaction value threshold
        """
        min_value = self.config.signals.get('min_transaction_value', 50000)

        # 10b5-1 plans are less informative
        if is_10b5_1:
            return 0

        # Below minimum threshold
        if trans_value < min_value:
            return 0

        title = owner_info.get('title', '').upper()
        is_purchase = trans_type == 'Purchase'
        is_sale = trans_type == 'Sale'

        # Determine role-based weight
        if 'CEO' in title or 'CHIEF EXECUTIVE' in title:
            base_weight = self.SIGNAL_WEIGHTS['CEO_BUY'] if is_purchase else self.SIGNAL_WEIGHTS['CEO_SELL']
        elif 'CFO' in title or 'CHIEF FINANCIAL' in title:
            base_weight = self.SIGNAL_WEIGHTS['CFO_BUY'] if is_purchase else self.SIGNAL_WEIGHTS['CFO_SELL']
        elif 'CMO' in title or 'CHIEF MEDICAL' in title:
            base_weight = self.SIGNAL_WEIGHTS['CMO_BUY'] if is_purchase else self.SIGNAL_WEIGHTS['CMO_SELL']
        elif owner_info.get('is_director'):
            base_weight = self.SIGNAL_WEIGHTS['DIRECTOR_BUY'] if is_purchase else -3
        elif owner_info.get('is_officer'):
            base_weight = self.SIGNAL_WEIGHTS['OFFICER_BUY'] if is_purchase else -3
        else:
            base_weight = 2 if is_purchase else -2

        # Scale by transaction value (larger = more significant)
        if trans_value > 500000:
            base_weight = int(base_weight * 1.5)
        elif trans_value > 200000:
            base_weight = int(base_weight * 1.2)

        return base_weight

    def scrape_company(self, ticker: str, days: int = 30) -> List[InsiderTransaction]:
        """
        Scrape all Form 4 filings for a company.

        Args:
            ticker: Company ticker symbol
            days: Number of days to look back

        Returns:
            List of InsiderTransaction objects
        """
        cik = self.get_cik_for_ticker(ticker)
        if not cik:
            logger.warning(f"Could not find CIK for ticker {ticker}")
            return []

        logger.info(f"Scraping Form 4 filings for {ticker} (CIK: {cik})")

        all_transactions = []
        filings = self.get_recent_form4_filings(cik, days)

        for filing in filings:
            # Get the XML URL from the filing page
            xml_url = self.get_form4_xml_url(filing['url'])
            if xml_url:
                transactions = self.parse_form4_xml(
                    xml_url, ticker, filing['filing_date']
                )
                all_transactions.extend(transactions)

        logger.info(f"Found {len(all_transactions)} transactions for {ticker}")
        return all_transactions

    def save_transactions(self, transactions: List[InsiderTransaction]) -> Tuple[int, int]:
        """
        Save transactions to the database.

        Returns:
            Tuple of (inserted_count, updated_count)
        """
        if not transactions:
            return 0, 0

        inserted = 0
        updated = 0

        for trans in transactions:
            data = {
                'company_ticker': trans.company_ticker,
                'company_cik': trans.company_cik,
                'insider_name': trans.insider_name,
                'insider_cik': trans.insider_cik,
                'insider_title': trans.insider_title,
                'is_director': trans.is_director,
                'is_officer': trans.is_officer,
                'is_ten_percent_owner': trans.is_ten_percent_owner,
                'transaction_date': trans.transaction_date,
                'transaction_type': trans.transaction_type,
                'transaction_code': trans.transaction_code,
                'shares': trans.shares,
                'price_per_share': trans.price_per_share,
                'transaction_value': trans.transaction_value,
                'shares_owned_after': trans.shares_owned_after,
                'ownership_nature': trans.ownership_nature,
                'is_10b5_1_plan': trans.is_10b5_1_plan,
                'footnotes': trans.footnotes,
                'filing_date': trans.filing_date,
                'filing_url': trans.filing_url,
                'signal_weight': trans.signal_weight,
            }

            try:
                # Try to insert (will fail on duplicate)
                self.db.insert('insider_transactions', data)
                inserted += 1
            except Exception:
                # Already exists
                pass

        logger.info(f"Saved {inserted} new transactions")
        return inserted, updated

    def run(self, tickers: Optional[List[str]] = None, days: int = 30) -> Dict[str, Any]:
        """
        Run the Form 4 scraper for all watchlist companies.

        Args:
            tickers: Optional list of tickers (defaults to watchlist)
            days: Number of days to look back

        Returns:
            Dictionary with run statistics
        """
        if tickers is None:
            tickers = self.config.watchlist

        run_id = self.db.log_scraper_run('form4')

        total_transactions = 0
        total_inserted = 0
        errors = []

        for ticker in tickers:
            try:
                transactions = self.scrape_company(ticker, days)
                inserted, _ = self.save_transactions(transactions)
                total_transactions += len(transactions)
                total_inserted += inserted
            except Exception as e:
                errors.append({'ticker': ticker, 'error': str(e)})
                logger.error(f"Failed to scrape {ticker}: {e}")

        self.db.update_scraper_run(
            run_id,
            status='completed' if not errors else 'completed_with_errors',
            records_processed=total_transactions,
            records_inserted=total_inserted,
            errors_count=len(errors),
            error_details={'errors': errors} if errors else None
        )

        return {
            'run_id': run_id,
            'tickers_processed': len(tickers),
            'transactions_found': total_transactions,
            'transactions_inserted': total_inserted,
            'errors': errors
        }


if __name__ == '__main__':
    # Test the scraper
    scraper = Form4Scraper()
    result = scraper.run(tickers=['MRNA', 'VRTX'], days=30)
    print(f"Scraper result: {result}")
