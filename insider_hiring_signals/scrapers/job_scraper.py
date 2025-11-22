"""
Job Posting Scraper for Biotech Companies
Uses company career pages and job aggregators since LinkedIn API requires approval
"""

import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from utils.config import get_config
from utils.database import get_database
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class JobPosting:
    """Represents a job posting."""
    company_ticker: str
    company_name: str
    job_title: str
    department: str
    seniority_level: str
    location: str
    employment_type: str
    description: str
    post_date: date
    first_seen_date: date
    source: str
    source_url: str
    is_senior_role: bool
    is_commercial_role: bool
    is_rd_role: bool
    is_clinical_role: bool
    is_manufacturing_role: bool
    signal_weight: int = 0


class JobScraper:
    """
    Scrapes job postings from various sources to detect hiring signals.

    Since LinkedIn API requires approval, this scraper uses:
    1. Company career pages (via Greenhouse, Lever, Workday, etc.)
    2. Indeed API/scraping (backup)
    3. Google Jobs API (if available)

    Key signals:
    - Commercial roles expansion: indicates upcoming product launch
    - Manufacturing scale-up: indicates production preparation
    - Clinical operations: indicates trial activity
    - Regulatory affairs: indicates FDA submission prep
    """

    # Department classification keywords
    DEPARTMENT_KEYWORDS = {
        'Commercial': [
            'commercial', 'sales', 'marketing', 'market access', 'business development',
            'account manager', 'territory', 'field sales', 'medical science liaison', 'msl',
            'brand manager', 'product manager', 'market research', 'pricing', 'reimbursement'
        ],
        'R&D': [
            'research', 'scientist', 'discovery', 'biology', 'chemistry', 'pharmacology',
            'toxicology', 'preclinical', 'translational', 'computational', 'bioinformatics',
            'genomics', 'proteomics', 'drug development', 'assay', 'screening'
        ],
        'Clinical': [
            'clinical', 'trial', 'cra', 'clinical research', 'medical monitor',
            'clinical operations', 'clinical project', 'clinical data', 'biostatistics',
            'medical writing', 'pharmacovigilance', 'drug safety', 'clinical supply'
        ],
        'Regulatory': [
            'regulatory', 'regulatory affairs', 'submission', 'cmc', 'quality',
            'compliance', 'gxp', 'validation', 'audit', 'fda', 'ema'
        ],
        'Manufacturing': [
            'manufacturing', 'production', 'process development', 'scale-up',
            'fill finish', 'formulation', 'supply chain', 'logistics', 'cmc',
            'technical operations', 'gmp', 'process engineer', 'bioprocess'
        ],
        'Medical Affairs': [
            'medical affairs', 'medical director', 'medical advisor', 'msl',
            'health economics', 'outcomes research', 'real world evidence',
            'publication', 'scientific communications'
        ],
        'Admin': [
            'human resources', 'hr', 'finance', 'accounting', 'legal', 'it',
            'facilities', 'administrative', 'executive assistant', 'receptionist'
        ]
    }

    # Seniority level keywords
    SENIORITY_KEYWORDS = {
        'C-Level': ['ceo', 'cfo', 'cmo', 'coo', 'cso', 'cto', 'chief'],
        'VP': ['vice president', 'vp ', 'svp', 'evp'],
        'Director': ['director', 'head of'],
        'Senior': ['senior', 'sr.', 'sr ', 'lead', 'principal', 'staff'],
        'Manager': ['manager', 'supervisor'],
        'Mid': ['associate', 'specialist', 'coordinator', 'analyst'],
        'Entry': ['junior', 'jr.', 'jr ', 'intern', 'entry', 'assistant']
    }

    # Signal weights for different job types
    SIGNAL_WEIGHTS = {
        'COMMERCIAL_ROLE': 5,
        'VP_MANUFACTURING': 4,
        'REGULATORY_EXPANSION': 3,
        'CLINICAL_TRIAL_ROLE': 4,
        'MSL_ROLE': 5,
        'SENIOR_HIRE': 3,
        'MASS_HIRING': 6,  # 5+ roles in same department
        'HIRING_FREEZE': -4,  # Job removals
    }

    # Known biotech career page patterns
    CAREER_PAGE_PATTERNS = {
        'greenhouse': 'boards.greenhouse.io',
        'lever': 'jobs.lever.co',
        'workday': 'myworkdayjobs.com',
        'smartrecruiters': 'jobs.smartrecruiters.com',
        'icims': 'careers-',  # Pattern varies
        'taleo': 'taleo.net',
    }

    # Company career page URLs (would be populated from config or database)
    COMPANY_CAREER_URLS = {
        'MRNA': 'https://boards.greenhouse.io/modernatx',
        'VRTX': 'https://www.vrtx.com/careers/open-positions',
        'REGN': 'https://careers.regeneron.com/search-jobs',
        'BIIB': 'https://www.biogen.com/careers.html',
        'ALNY': 'https://www.alnylam.com/careers',
        'BMRN': 'https://careers.biomarin.com',
        'INCY': 'https://www.incyte.com/careers',
        'CRSP': 'https://boards.greenhouse.io/crisprtx',
        'BEAM': 'https://boards.greenhouse.io/beamtx',
        'EDIT': 'https://www.editasmedicine.com/careers/',
        'NTLA': 'https://www.intelliatx.com/careers/',
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.db = get_database(config_path)
        self.session = self._create_session()
        self.rate_limit_seconds = self.config.scraping.get('rate_limit_seconds', 2.0)
        self.last_request_time = 0

    def _create_session(self) -> requests.Session:
        """Create a requests session with browser-like headers."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        return session

    def _rate_limit_wait(self):
        """Ensure we respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Dict = None) -> Optional[requests.Response]:
        """Make a rate-limited request."""
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

    def classify_job(self, title: str, description: str = '') -> Dict[str, Any]:
        """
        Classify a job posting by department and seniority.

        Args:
            title: Job title
            description: Job description (optional)

        Returns:
            Dictionary with classification results
        """
        title_lower = title.lower()
        desc_lower = description.lower() if description else ''
        combined = f"{title_lower} {desc_lower}"

        # Determine department
        department = 'Other'
        dept_scores = {}

        for dept, keywords in self.DEPARTMENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > 0:
                dept_scores[dept] = score

        if dept_scores:
            department = max(dept_scores, key=dept_scores.get)

        # Determine seniority
        seniority = 'Mid'  # Default

        for level, keywords in self.SENIORITY_KEYWORDS.items():
            if any(kw in title_lower for kw in keywords):
                seniority = level
                break

        # Determine role flags
        is_commercial = department in ['Commercial', 'Medical Affairs']
        is_rd = department == 'R&D'
        is_clinical = department == 'Clinical'
        is_manufacturing = department == 'Manufacturing'
        is_senior = seniority in ['C-Level', 'VP', 'Director', 'Senior']

        # Calculate signal weight
        signal_weight = 0

        if is_commercial:
            signal_weight += self.SIGNAL_WEIGHTS['COMMERCIAL_ROLE']
        if is_manufacturing and is_senior:
            signal_weight += self.SIGNAL_WEIGHTS['VP_MANUFACTURING']
        if department == 'Regulatory':
            signal_weight += self.SIGNAL_WEIGHTS['REGULATORY_EXPANSION']
        if is_clinical:
            signal_weight += self.SIGNAL_WEIGHTS['CLINICAL_TRIAL_ROLE']
        if 'msl' in title_lower or 'medical science liaison' in title_lower:
            signal_weight += self.SIGNAL_WEIGHTS['MSL_ROLE']
        if is_senior and signal_weight == 0:
            signal_weight += self.SIGNAL_WEIGHTS['SENIOR_HIRE']

        return {
            'department': department,
            'seniority_level': seniority,
            'is_commercial_role': is_commercial,
            'is_rd_role': is_rd,
            'is_clinical_role': is_clinical,
            'is_manufacturing_role': is_manufacturing,
            'is_senior_role': is_senior,
            'signal_weight': signal_weight
        }

    def scrape_greenhouse(self, company_ticker: str, greenhouse_url: str) -> List[JobPosting]:
        """
        Scrape jobs from Greenhouse career page.

        Greenhouse provides a JSON API at /departments endpoint.
        """
        jobs = []

        try:
            # Get company name from board URL
            board_token = greenhouse_url.split('/')[-1]
            api_url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs"

            response = self._make_request(api_url)
            if not response:
                return jobs

            data = response.json()
            company_name = data.get('name', company_ticker)

            for job_data in data.get('jobs', []):
                title = job_data.get('title', '')
                location = job_data.get('location', {}).get('name', '')
                job_url = job_data.get('absolute_url', '')
                updated_at = job_data.get('updated_at', '')

                # Parse date
                post_date = date.today()
                if updated_at:
                    try:
                        post_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00')).date()
                    except ValueError:
                        pass

                # Get job description if available
                description = ''
                content = job_data.get('content', '')
                if content:
                    soup = BeautifulSoup(content, 'html.parser')
                    description = soup.get_text(separator=' ', strip=True)[:1000]

                # Classify the job
                classification = self.classify_job(title, description)

                jobs.append(JobPosting(
                    company_ticker=company_ticker,
                    company_name=company_name,
                    job_title=title,
                    department=classification['department'],
                    seniority_level=classification['seniority_level'],
                    location=location,
                    employment_type='Full-time',
                    description=description,
                    post_date=post_date,
                    first_seen_date=date.today(),
                    source='Greenhouse',
                    source_url=job_url,
                    is_senior_role=classification['is_senior_role'],
                    is_commercial_role=classification['is_commercial_role'],
                    is_rd_role=classification['is_rd_role'],
                    is_clinical_role=classification['is_clinical_role'],
                    is_manufacturing_role=classification['is_manufacturing_role'],
                    signal_weight=classification['signal_weight']
                ))

            logger.info(f"Found {len(jobs)} jobs from Greenhouse for {company_ticker}")

        except Exception as e:
            logger.error(f"Failed to scrape Greenhouse for {company_ticker}: {e}")

        return jobs

    def scrape_lever(self, company_ticker: str, lever_url: str) -> List[JobPosting]:
        """Scrape jobs from Lever career page."""
        jobs = []

        try:
            # Lever has a JSON API
            board_name = lever_url.split('/')[-1]
            api_url = f"https://api.lever.co/v0/postings/{board_name}"

            response = self._make_request(api_url)
            if not response:
                return jobs

            data = response.json()

            for job_data in data:
                title = job_data.get('text', '')
                location = job_data.get('categories', {}).get('location', '')
                team = job_data.get('categories', {}).get('team', '')
                job_url = job_data.get('hostedUrl', '')

                # Get posting date
                created_at = job_data.get('createdAt', 0)
                post_date = datetime.fromtimestamp(created_at / 1000).date() if created_at else date.today()

                # Get description
                description = ''
                for list_item in job_data.get('lists', []):
                    description += list_item.get('text', '') + ' '
                description = description[:1000]

                # Classify
                classification = self.classify_job(title, description)

                jobs.append(JobPosting(
                    company_ticker=company_ticker,
                    company_name=company_ticker,
                    job_title=title,
                    department=classification['department'],
                    seniority_level=classification['seniority_level'],
                    location=location,
                    employment_type='Full-time',
                    description=description,
                    post_date=post_date,
                    first_seen_date=date.today(),
                    source='Lever',
                    source_url=job_url,
                    is_senior_role=classification['is_senior_role'],
                    is_commercial_role=classification['is_commercial_role'],
                    is_rd_role=classification['is_rd_role'],
                    is_clinical_role=classification['is_clinical_role'],
                    is_manufacturing_role=classification['is_manufacturing_role'],
                    signal_weight=classification['signal_weight']
                ))

            logger.info(f"Found {len(jobs)} jobs from Lever for {company_ticker}")

        except Exception as e:
            logger.error(f"Failed to scrape Lever for {company_ticker}: {e}")

        return jobs

    def scrape_generic_career_page(self, company_ticker: str, url: str) -> List[JobPosting]:
        """
        Scrape jobs from a generic career page using HTML parsing.

        This is a fallback when APIs aren't available.
        """
        jobs = []

        try:
            response = self._make_request(url)
            if not response:
                return jobs

            soup = BeautifulSoup(response.text, 'html.parser')

            # Common job listing patterns
            job_selectors = [
                'div.job-listing', 'div.job-item', 'div.position',
                'li.job', 'article.job', 'tr.job-row',
                '[data-job]', '[class*="job"]', '[class*="position"]',
                'a[href*="job"]', 'a[href*="career"]', 'a[href*="position"]'
            ]

            job_elements = []
            for selector in job_selectors:
                elements = soup.select(selector)
                if elements:
                    job_elements = elements
                    break

            # If no structured jobs found, look for links with job-related keywords
            if not job_elements:
                job_links = soup.find_all('a', href=True)
                for link in job_links:
                    href = link.get('href', '').lower()
                    text = link.get_text().strip()

                    # Check if this looks like a job link
                    if any(kw in href for kw in ['job', 'career', 'position', 'apply']):
                        if len(text) > 5 and len(text) < 200:
                            classification = self.classify_job(text)

                            jobs.append(JobPosting(
                                company_ticker=company_ticker,
                                company_name=company_ticker,
                                job_title=text,
                                department=classification['department'],
                                seniority_level=classification['seniority_level'],
                                location='',
                                employment_type='Full-time',
                                description='',
                                post_date=date.today(),
                                first_seen_date=date.today(),
                                source='Company Website',
                                source_url=urljoin(url, link.get('href', '')),
                                is_senior_role=classification['is_senior_role'],
                                is_commercial_role=classification['is_commercial_role'],
                                is_rd_role=classification['is_rd_role'],
                                is_clinical_role=classification['is_clinical_role'],
                                is_manufacturing_role=classification['is_manufacturing_role'],
                                signal_weight=classification['signal_weight']
                            ))

            logger.info(f"Found {len(jobs)} jobs from generic scrape for {company_ticker}")

        except Exception as e:
            logger.error(f"Failed to scrape career page for {company_ticker}: {e}")

        return jobs

    def scrape_company(self, company_ticker: str,
                       career_url: Optional[str] = None) -> List[JobPosting]:
        """
        Scrape job postings for a single company.

        Args:
            company_ticker: Company ticker symbol
            career_url: Optional career page URL

        Returns:
            List of JobPosting objects
        """
        if not career_url:
            career_url = self.COMPANY_CAREER_URLS.get(company_ticker)

        if not career_url:
            logger.warning(f"No career page URL for {company_ticker}")
            return []

        logger.info(f"Scraping jobs for {company_ticker} from {career_url}")

        # Determine the platform and use appropriate scraper
        if 'greenhouse.io' in career_url:
            return self.scrape_greenhouse(company_ticker, career_url)
        elif 'lever.co' in career_url:
            return self.scrape_lever(company_ticker, career_url)
        else:
            return self.scrape_generic_career_page(company_ticker, career_url)

    def detect_job_removals(self, company_ticker: str,
                            current_jobs: List[JobPosting]) -> List[Dict]:
        """
        Detect jobs that were previously seen but are now gone.

        Returns list of removed job info for signal generation.
        """
        # Get previous job postings for this company that haven't been marked as removed
        previous_jobs = self.db.execute("""
            SELECT job_id, job_title, department, source_url, first_seen_date
            FROM job_postings
            WHERE company_ticker = %s
            AND removal_date IS NULL
            AND first_seen_date >= %s
        """, (company_ticker, date.today() - timedelta(days=90)))

        current_urls = {job.source_url for job in current_jobs}
        removals = []

        for prev_job in previous_jobs:
            if prev_job['source_url'] not in current_urls:
                # Mark as removed
                self.db.execute("""
                    UPDATE job_postings
                    SET removal_date = %s, last_seen_date = %s
                    WHERE job_id = %s
                """, (date.today(), date.today(), prev_job['job_id']))

                removals.append({
                    'job_title': prev_job['job_title'],
                    'department': prev_job['department'],
                    'days_active': (date.today() - prev_job['first_seen_date']).days
                })

        if removals:
            logger.info(f"Detected {len(removals)} job removals for {company_ticker}")

        return removals

    def calculate_hiring_signals(self, company_ticker: str,
                                  jobs: List[JobPosting]) -> Dict[str, Any]:
        """
        Calculate aggregate hiring signals for a company.

        Returns signal summary with weights.
        """
        signals = {
            'commercial_count': 0,
            'rd_count': 0,
            'clinical_count': 0,
            'manufacturing_count': 0,
            'senior_count': 0,
            'total_count': len(jobs),
            'aggregate_weight': 0,
            'signal_type': 'NEUTRAL'
        }

        for job in jobs:
            if job.is_commercial_role:
                signals['commercial_count'] += 1
            if job.is_rd_role:
                signals['rd_count'] += 1
            if job.is_clinical_role:
                signals['clinical_count'] += 1
            if job.is_manufacturing_role:
                signals['manufacturing_count'] += 1
            if job.is_senior_role:
                signals['senior_count'] += 1

            signals['aggregate_weight'] += job.signal_weight

        # Determine primary signal type
        if signals['commercial_count'] >= 5:
            signals['signal_type'] = 'COMMERCIAL_BUILDUP'
            signals['aggregate_weight'] += self.SIGNAL_WEIGHTS['MASS_HIRING']
        elif signals['manufacturing_count'] >= 3:
            signals['signal_type'] = 'MANUFACTURING_SCALEUP'
        elif signals['clinical_count'] >= 3:
            signals['signal_type'] = 'CLINICAL_EXPANSION'
        elif signals['total_count'] >= 10:
            signals['signal_type'] = 'GENERAL_EXPANSION'

        return signals

    def save_jobs(self, jobs: List[JobPosting]) -> Tuple[int, int]:
        """
        Save job postings to database.

        Returns:
            Tuple of (inserted_count, updated_count)
        """
        if not jobs:
            return 0, 0

        inserted = 0
        updated = 0

        for job in jobs:
            data = {
                'company_ticker': job.company_ticker,
                'company_name': job.company_name,
                'job_title': job.job_title,
                'department': job.department,
                'seniority_level': job.seniority_level,
                'location': job.location,
                'employment_type': job.employment_type,
                'description': job.description[:2000] if job.description else '',
                'post_date': job.post_date,
                'first_seen_date': job.first_seen_date,
                'last_seen_date': date.today(),
                'source': job.source,
                'source_url': job.source_url,
                'is_senior_role': job.is_senior_role,
                'is_commercial_role': job.is_commercial_role,
                'is_rd_role': job.is_rd_role,
                'is_clinical_role': job.is_clinical_role,
                'is_manufacturing_role': job.is_manufacturing_role,
                'signal_weight': job.signal_weight,
            }

            try:
                # Try to upsert
                result = self.db.upsert(
                    'job_postings',
                    data,
                    conflict_columns=['company_ticker', 'job_title', 'source', 'post_date'],
                    update_columns=['last_seen_date', 'description', 'location']
                )
                if result:
                    inserted += 1
            except Exception as e:
                logger.error(f"Failed to save job: {e}")

        logger.info(f"Saved {inserted} jobs")
        return inserted, updated

    def run(self, tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the job scraper for all watchlist companies.

        Args:
            tickers: Optional list of tickers (defaults to watchlist)

        Returns:
            Dictionary with run statistics
        """
        if tickers is None:
            tickers = self.config.watchlist

        run_id = self.db.log_scraper_run('jobs')

        total_jobs = 0
        total_inserted = 0
        company_signals = {}
        errors = []

        for ticker in tickers:
            try:
                # Skip if no career URL configured
                if ticker not in self.COMPANY_CAREER_URLS:
                    continue

                jobs = self.scrape_company(ticker)
                inserted, _ = self.save_jobs(jobs)

                # Detect removals
                removals = self.detect_job_removals(ticker, jobs)

                # Calculate signals
                signals = self.calculate_hiring_signals(ticker, jobs)
                signals['removals'] = len(removals)

                total_jobs += len(jobs)
                total_inserted += inserted
                company_signals[ticker] = signals

            except Exception as e:
                errors.append({'ticker': ticker, 'error': str(e)})
                logger.error(f"Failed to scrape jobs for {ticker}: {e}")

        self.db.update_scraper_run(
            run_id,
            status='completed' if not errors else 'completed_with_errors',
            records_processed=total_jobs,
            records_inserted=total_inserted,
            errors_count=len(errors),
            error_details={
                'errors': errors,
                'company_signals': company_signals
            } if errors else {'company_signals': company_signals}
        )

        return {
            'run_id': run_id,
            'companies_processed': len([t for t in tickers if t in self.COMPANY_CAREER_URLS]),
            'jobs_found': total_jobs,
            'jobs_inserted': total_inserted,
            'company_signals': company_signals,
            'errors': errors
        }


if __name__ == '__main__':
    # Test the scraper
    scraper = JobScraper()

    # Test with a few companies
    result = scraper.run(tickers=['MRNA', 'CRSP', 'BEAM'])
    print(f"Scraper result: {result}")
