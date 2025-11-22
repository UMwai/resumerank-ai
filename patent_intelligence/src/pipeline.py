"""
Patent Intelligence ETL Pipeline

Main orchestrator for the patent intelligence data pipeline.
Coordinates extraction, transformation, and loading of patent cliff data.
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .extractors.orange_book import OrangeBookExtractor
from .extractors.uspto import USPTOExtractor, PatentExpirationCalculator
from .extractors.fda_anda import ANDAExtractor, GenericCompetitionAnalyzer
from .loaders.db_loader import DatabaseLoader
from .transformers.scoring import PatentCliffScorer, DrugPatentData
from .transformers.calendar import PatentCliffCalendarGenerator
from .utils.config import Config, get_config
from .utils.email import EmailNotifier
from .utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


class PatentIntelligencePipeline:
    """
    Main ETL pipeline for the Patent Intelligence system.

    Orchestrates:
    1. Data extraction from FDA Orange Book, USPTO, and ANDA sources
    2. Patent expiration calculations and scoring
    3. Calendar generation and database loading
    4. Email notifications
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to configuration file.
        """
        self.config = get_config(config_path)

        # Setup logging
        setup_logger(
            log_level=self.config.get("logging.level", "INFO"),
            log_file=self.config.get("logging.log_file"),
        )

        # Initialize extractors
        self.orange_book_extractor = OrangeBookExtractor(
            cache_dir=self.config.get("cache.cache_dir", ".cache/orange_book"),
            use_cache=self.config.get("cache.enabled", True),
            cache_ttl_hours=self.config.get("cache.ttl_hours", 24),
        )

        self.uspto_extractor = USPTOExtractor(
            api_key=self.config.get("data_sources.uspto.api_key"),
        )

        self.anda_extractor = ANDAExtractor(
            cache_dir=self.config.get("cache.cache_dir", ".cache/anda"),
        )

        # Initialize transformers
        self.scorer = PatentCliffScorer()
        self.calendar_generator = PatentCliffCalendarGenerator(
            forward_months=self.config.get("calendar.forward_months", 12),
        )

        # Database loader (initialized when needed)
        self._db_loader: Optional[DatabaseLoader] = None

        # Email notifier (initialized when needed)
        self._email_notifier: Optional[EmailNotifier] = None

        # Data storage
        self.drugs: List[Dict] = []
        self.patents: List[Dict] = []
        self.andas: List[Dict] = []
        self.calendar_events: List[Dict] = []

    @property
    def db_loader(self) -> DatabaseLoader:
        """Get database loader, initializing if needed."""
        if self._db_loader is None:
            db_config = self.config.get_database_config()
            self._db_loader = DatabaseLoader(db_config)
        return self._db_loader

    @property
    def email_notifier(self) -> EmailNotifier:
        """Get email notifier, initializing if needed."""
        if self._email_notifier is None:
            email_config = self.config.get_email_config()
            self._email_notifier = EmailNotifier(
                smtp_server=email_config["smtp_server"],
                smtp_port=email_config["smtp_port"],
                sender_email=email_config["sender_email"],
                sender_password=email_config["sender_password"],
                use_tls=email_config["use_tls"],
            )
        return self._email_notifier

    def extract_orange_book_data(self, top_n: int = 50) -> None:
        """
        Extract drug and patent data from FDA Orange Book.

        Args:
            top_n: Number of top drugs to extract.
        """
        logger.info(f"Extracting Orange Book data (top {top_n} drugs)...")

        try:
            drugs, patents, exclusivity = self.orange_book_extractor.extract_for_database(
                top_n=top_n
            )

            self.drugs = drugs
            self.patents = patents

            logger.info(
                f"Extracted {len(self.drugs)} drugs and {len(self.patents)} patents"
            )

        except Exception as e:
            logger.error(f"Orange Book extraction failed: {e}")
            raise

    def enrich_patent_data(self) -> None:
        """
        Enrich patent data with USPTO information and calculations.
        """
        logger.info("Enriching patent data with USPTO information...")

        calculator = PatentExpirationCalculator()

        for patent in self.patents:
            # Calculate expiration dates
            base_exp = patent.get("base_expiration_date")

            if base_exp:
                expirations = calculator.calculate_expiration(
                    orange_book_expiration=base_exp,
                    pta_days=patent.get("pta_days", 0),
                    pte_days=patent.get("pte_days", 0),
                )

                patent["adjusted_expiration_date"] = expirations.get(
                    "pte_adjusted_date"
                ) or expirations.get("pta_adjusted_date")
                patent["final_expiration_date"] = expirations.get(
                    "final_expiration_date"
                )

        logger.info("Patent data enriched with expiration calculations")

    def extract_anda_data(self) -> None:
        """
        Extract ANDA (generic application) data.
        """
        logger.info("Extracting ANDA data...")

        # Get active ingredients from extracted drugs
        ingredients = []
        for drug in self.drugs:
            ingredient = drug.get("active_ingredient") or drug.get("generic_name")
            if ingredient:
                ingredients.append(ingredient)

        # Extract ANDAs
        self.andas = self.anda_extractor.extract_for_database(
            active_ingredients=ingredients
        )

        logger.info(f"Extracted {len(self.andas)} ANDA records")

    def generate_calendar(self) -> None:
        """
        Generate patent cliff calendar from extracted data.
        """
        logger.info("Generating patent cliff calendar...")

        events = self.calendar_generator.generate_from_raw_data(
            drugs=self.drugs,
            patents=self.patents,
            andas=self.andas,
        )

        # Convert to dictionaries for storage
        self.calendar_events = self.calendar_generator.get_events_for_database(events)

        logger.info(f"Generated {len(self.calendar_events)} calendar events")

    def load_to_database(self) -> Dict[str, int]:
        """
        Load all extracted and transformed data to database.

        Returns:
            Dictionary with counts of records loaded.
        """
        logger.info("Loading data to database...")

        stats = {}

        # Load drugs
        inserted, updated = self.db_loader.load_drugs(self.drugs)
        stats["drugs_inserted"] = inserted
        stats["drugs_updated"] = updated

        # Load patents
        inserted, updated = self.db_loader.load_patents(self.patents)
        stats["patents_inserted"] = inserted
        stats["patents_updated"] = updated

        # Load ANDAs
        inserted, updated = self.db_loader.load_generic_applications(self.andas)
        stats["andas_inserted"] = inserted
        stats["andas_updated"] = updated

        # Load calendar events
        inserted, updated = self.db_loader.load_calendar_events(self.calendar_events)
        stats["events_inserted"] = inserted
        stats["events_updated"] = updated

        logger.info(f"Database load complete: {stats}")
        return stats

    def send_weekly_digest(self) -> bool:
        """
        Send weekly digest email.

        Returns:
            True if email sent successfully.
        """
        email_config = self.config.get_email_config()

        if not email_config.get("enabled"):
            logger.info("Email notifications disabled")
            return False

        recipients = email_config.get("recipients", [])
        if not recipients:
            logger.warning("No email recipients configured")
            return False

        # Convert calendar events to format expected by email notifier
        events_for_email = []
        for event in self.calendar_events:
            events_for_email.append({
                "brand_name": next(
                    (d["brand_name"] for d in self.drugs if d.get("drug_id") == event.get("drug_id")),
                    "Unknown",
                ),
                "generic_name": next(
                    (d["generic_name"] for d in self.drugs if d.get("drug_id") == event.get("drug_id")),
                    "Unknown",
                ),
                "branded_company": next(
                    (d["branded_company"] for d in self.drugs if d.get("drug_id") == event.get("drug_id")),
                    "Unknown",
                ),
                "branded_company_ticker": next(
                    (d.get("branded_company_ticker") for d in self.drugs if d.get("drug_id") == event.get("drug_id")),
                    None,
                ),
                **event,
            })

        return self.email_notifier.send_weekly_digest(
            recipients=recipients,
            events=events_for_email,
        )

    def export_calendar(self, output_dir: str = "output") -> Dict[str, str]:
        """
        Export calendar to various formats.

        Args:
            output_dir: Directory for output files.

        Returns:
            Dictionary of output file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate events
        events = self.calendar_generator.generate_from_raw_data(
            drugs=self.drugs,
            patents=self.patents,
            andas=self.andas,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}

        # Export CSV
        csv_path = output_path / f"patent_cliff_calendar_{timestamp}.csv"
        self.calendar_generator.export_csv(events, str(csv_path))
        files["csv"] = str(csv_path)

        # Export JSON
        json_path = output_path / f"patent_cliff_calendar_{timestamp}.json"
        self.calendar_generator.export_json(events, str(json_path))
        files["json"] = str(json_path)

        # Export text report
        report = self.calendar_generator.format_calendar_report(events)
        report_path = output_path / f"patent_cliff_report_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        files["report"] = str(report_path)

        logger.info(f"Calendar exported to {output_dir}")
        return files

    def run_full_pipeline(
        self,
        top_n: int = 50,
        load_db: bool = True,
        send_email: bool = False,
        export_files: bool = True,
        output_dir: str = "output",
    ) -> Dict[str, Any]:
        """
        Run the complete ETL pipeline.

        Args:
            top_n: Number of top drugs to process.
            load_db: Whether to load data to database.
            send_email: Whether to send email digest.
            export_files: Whether to export calendar files.
            output_dir: Directory for output files.

        Returns:
            Dictionary with pipeline execution results.
        """
        logger.info("Starting Patent Intelligence Pipeline...")
        start_time = datetime.now()

        results = {
            "start_time": start_time.isoformat(),
            "status": "SUCCESS",
            "errors": [],
        }

        try:
            # Step 1: Extract Orange Book data
            self.extract_orange_book_data(top_n=top_n)
            results["drugs_extracted"] = len(self.drugs)
            results["patents_extracted"] = len(self.patents)

            # Step 2: Enrich patent data
            self.enrich_patent_data()

            # Step 3: Extract ANDA data
            self.extract_anda_data()
            results["andas_extracted"] = len(self.andas)

            # Step 4: Generate calendar
            self.generate_calendar()
            results["events_generated"] = len(self.calendar_events)

            # Step 5: Load to database (if enabled)
            if load_db:
                try:
                    db_stats = self.load_to_database()
                    results["database"] = db_stats
                except Exception as e:
                    logger.error(f"Database load failed: {e}")
                    results["errors"].append(f"Database load: {str(e)}")

            # Step 6: Export files (if enabled)
            if export_files:
                try:
                    files = self.export_calendar(output_dir)
                    results["exported_files"] = files
                except Exception as e:
                    logger.error(f"File export failed: {e}")
                    results["errors"].append(f"File export: {str(e)}")

            # Step 7: Send email (if enabled)
            if send_email:
                try:
                    email_sent = self.send_weekly_digest()
                    results["email_sent"] = email_sent
                except Exception as e:
                    logger.error(f"Email send failed: {e}")
                    results["errors"].append(f"Email send: {str(e)}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results["status"] = "FAILED"
            results["errors"].append(str(e))

        end_time = datetime.now()
        results["end_time"] = end_time.isoformat()
        results["duration_seconds"] = (end_time - start_time).total_seconds()

        logger.info(f"Pipeline completed in {results['duration_seconds']:.1f} seconds")
        logger.info(f"Results: {results}")

        return results

    def close(self) -> None:
        """Clean up resources."""
        if self._db_loader:
            self._db_loader.close()


def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Patent Intelligence ETL Pipeline"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--top-drugs",
        type=int,
        default=50,
        help="Number of top drugs to process (default: 50)",
    )

    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip database loading",
    )

    parser.add_argument(
        "--send-email",
        action="store_true",
        help="Send weekly digest email",
    )

    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip file export",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for exported files",
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = PatentIntelligencePipeline(config_path=args.config)

    try:
        results = pipeline.run_full_pipeline(
            top_n=args.top_drugs,
            load_db=not args.no_db,
            send_email=args.send_email,
            export_files=not args.no_export,
            output_dir=args.output_dir,
        )

        if results["status"] == "SUCCESS":
            print("\nPipeline completed successfully!")
            print(f"Duration: {results['duration_seconds']:.1f} seconds")
            print(f"Drugs processed: {results.get('drugs_extracted', 0)}")
            print(f"Patents processed: {results.get('patents_extracted', 0)}")
            print(f"Calendar events: {results.get('events_generated', 0)}")

            if "exported_files" in results:
                print("\nExported files:")
                for file_type, path in results["exported_files"].items():
                    print(f"  {file_type}: {path}")

            sys.exit(0)
        else:
            print("\nPipeline failed!")
            for error in results.get("errors", []):
                print(f"  - {error}")
            sys.exit(1)

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
