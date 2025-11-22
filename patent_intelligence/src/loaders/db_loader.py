"""
Database Loader for Patent Intelligence System

Handles loading extracted and transformed data into PostgreSQL database.
Implements upsert logic to handle incremental updates.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from ..utils.database import DatabaseConnection, get_database
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseLoader:
    """
    Loads patent intelligence data into PostgreSQL database.

    Handles:
    - Drug records from Orange Book
    - Patent information with expiration dates
    - ANDA (generic application) records
    - Litigation tracking
    - Patent cliff calendar events
    """

    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the database loader.

        Args:
            db_config: Database configuration dictionary.
        """
        if db_config:
            self.db = DatabaseConnection(**db_config)
        else:
            self.db = DatabaseConnection()

        self.db.initialize_pool()

    def load_drugs(self, drugs: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Load drug records into the database.

        Args:
            drugs: List of drug dictionaries.

        Returns:
            Tuple of (inserted_count, updated_count).
        """
        inserted = 0
        updated = 0

        for drug in drugs:
            try:
                # Check if drug exists by NDA number
                existing = self.db.execute_query(
                    "SELECT drug_id FROM drugs WHERE nda_number = %s",
                    (drug.get("nda_number"),),
                )

                if existing:
                    # Update existing record
                    result = self.db.execute_write(
                        """
                        UPDATE drugs SET
                            brand_name = %s,
                            generic_name = %s,
                            active_ingredient = %s,
                            branded_company = %s,
                            branded_company_ticker = %s,
                            therapeutic_area = %s,
                            dosage_form = %s,
                            route_of_administration = %s,
                            annual_revenue = %s,
                            revenue_year = %s,
                            fda_approval_date = %s,
                            market_status = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE nda_number = %s
                        """,
                        (
                            drug.get("brand_name"),
                            drug.get("generic_name"),
                            drug.get("active_ingredient"),
                            drug.get("branded_company"),
                            drug.get("branded_company_ticker"),
                            drug.get("therapeutic_area"),
                            drug.get("dosage_form"),
                            drug.get("route_of_administration"),
                            drug.get("annual_revenue"),
                            drug.get("revenue_year"),
                            drug.get("fda_approval_date"),
                            drug.get("market_status", "ACTIVE"),
                            drug.get("nda_number"),
                        ),
                    )
                    updated += 1
                else:
                    # Insert new record
                    self.db.execute_write(
                        """
                        INSERT INTO drugs (
                            nda_number, brand_name, generic_name, active_ingredient,
                            branded_company, branded_company_ticker, therapeutic_area,
                            dosage_form, route_of_administration, annual_revenue,
                            revenue_year, fda_approval_date, market_status
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            drug.get("nda_number"),
                            drug.get("brand_name"),
                            drug.get("generic_name"),
                            drug.get("active_ingredient"),
                            drug.get("branded_company"),
                            drug.get("branded_company_ticker"),
                            drug.get("therapeutic_area"),
                            drug.get("dosage_form"),
                            drug.get("route_of_administration"),
                            drug.get("annual_revenue"),
                            drug.get("revenue_year"),
                            drug.get("fda_approval_date"),
                            drug.get("market_status", "ACTIVE"),
                        ),
                    )
                    inserted += 1

            except Exception as e:
                logger.error(f"Error loading drug {drug.get('nda_number')}: {e}")

        logger.info(f"Drugs loaded: {inserted} inserted, {updated} updated")
        return inserted, updated

    def load_patents(self, patents: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Load patent records into the database.

        Args:
            patents: List of patent dictionaries.

        Returns:
            Tuple of (inserted_count, updated_count).
        """
        inserted = 0
        updated = 0

        for patent in patents:
            try:
                # Get drug_id from NDA number
                drug_result = self.db.execute_query(
                    "SELECT drug_id FROM drugs WHERE nda_number = %s",
                    (patent.get("nda_number"),),
                )

                if not drug_result:
                    logger.warning(
                        f"Drug not found for patent {patent.get('patent_number')}, "
                        f"NDA: {patent.get('nda_number')}"
                    )
                    continue

                drug_id = drug_result[0]["drug_id"]

                # Check if patent exists
                existing = self.db.execute_query(
                    "SELECT patent_id FROM patents WHERE patent_number = %s",
                    (patent.get("patent_number"),),
                )

                # Calculate final expiration date
                base_exp = patent.get("base_expiration_date")
                adjusted_exp = patent.get("adjusted_expiration_date")
                pediatric_exp = patent.get("pediatric_exclusivity_date")

                final_exp = pediatric_exp or adjusted_exp or base_exp

                if existing:
                    # Update existing
                    self.db.execute_write(
                        """
                        UPDATE patents SET
                            drug_id = %s,
                            patent_type = %s,
                            patent_use_code = %s,
                            patent_claims = %s,
                            filing_date = %s,
                            grant_date = %s,
                            base_expiration_date = %s,
                            pta_days = %s,
                            pte_days = %s,
                            adjusted_expiration_date = %s,
                            pediatric_exclusivity_date = %s,
                            final_expiration_date = %s,
                            exclusivity_code = %s,
                            exclusivity_expiration = %s,
                            patent_status = %s,
                            strength_score = %s,
                            data_source = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE patent_number = %s
                        """,
                        (
                            drug_id,
                            patent.get("patent_type"),
                            patent.get("patent_use_code"),
                            patent.get("patent_claims"),
                            patent.get("filing_date"),
                            patent.get("grant_date"),
                            base_exp,
                            patent.get("pta_days", 0),
                            patent.get("pte_days", 0),
                            adjusted_exp,
                            pediatric_exp,
                            final_exp,
                            patent.get("exclusivity_code"),
                            patent.get("exclusivity_expiration"),
                            patent.get("patent_status", "ACTIVE"),
                            patent.get("strength_score"),
                            patent.get("data_source", "ORANGE_BOOK"),
                            patent.get("patent_number"),
                        ),
                    )
                    updated += 1
                else:
                    # Insert new
                    self.db.execute_write(
                        """
                        INSERT INTO patents (
                            patent_number, drug_id, patent_type, patent_use_code,
                            patent_claims, filing_date, grant_date, base_expiration_date,
                            pta_days, pte_days, adjusted_expiration_date,
                            pediatric_exclusivity_date, final_expiration_date,
                            exclusivity_code, exclusivity_expiration, patent_status,
                            strength_score, data_source
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            patent.get("patent_number"),
                            drug_id,
                            patent.get("patent_type"),
                            patent.get("patent_use_code"),
                            patent.get("patent_claims"),
                            patent.get("filing_date"),
                            patent.get("grant_date"),
                            base_exp,
                            patent.get("pta_days", 0),
                            patent.get("pte_days", 0),
                            adjusted_exp,
                            pediatric_exp,
                            final_exp,
                            patent.get("exclusivity_code"),
                            patent.get("exclusivity_expiration"),
                            patent.get("patent_status", "ACTIVE"),
                            patent.get("strength_score"),
                            patent.get("data_source", "ORANGE_BOOK"),
                        ),
                    )
                    inserted += 1

            except Exception as e:
                logger.error(f"Error loading patent {patent.get('patent_number')}: {e}")

        logger.info(f"Patents loaded: {inserted} inserted, {updated} updated")
        return inserted, updated

    def load_generic_applications(
        self, andas: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """
        Load ANDA (generic application) records.

        Args:
            andas: List of ANDA dictionaries.

        Returns:
            Tuple of (inserted_count, updated_count).
        """
        inserted = 0
        updated = 0

        for anda in andas:
            try:
                # Try to find matching drug by active ingredient
                drug_result = None
                if anda.get("reference_nda"):
                    drug_result = self.db.execute_query(
                        "SELECT drug_id FROM drugs WHERE nda_number = %s",
                        (anda.get("reference_nda"),),
                    )

                if not drug_result and anda.get("active_ingredient"):
                    drug_result = self.db.execute_query(
                        """
                        SELECT drug_id FROM drugs
                        WHERE UPPER(active_ingredient) LIKE UPPER(%s)
                        OR UPPER(generic_name) LIKE UPPER(%s)
                        LIMIT 1
                        """,
                        (
                            f"%{anda.get('active_ingredient')}%",
                            f"%{anda.get('active_ingredient')}%",
                        ),
                    )

                drug_id = drug_result[0]["drug_id"] if drug_result else None

                # Check if ANDA exists
                existing = self.db.execute_query(
                    "SELECT anda_id FROM generic_applications WHERE anda_number = %s",
                    (anda.get("anda_number"),),
                )

                if existing:
                    # Update
                    self.db.execute_write(
                        """
                        UPDATE generic_applications SET
                            drug_id = %s,
                            generic_company = %s,
                            generic_company_ticker = %s,
                            generic_drug_name = %s,
                            dosage_form = %s,
                            strength = %s,
                            filing_date = %s,
                            first_to_file = %s,
                            paragraph_iv_certification = %s,
                            tentative_approval_date = %s,
                            final_approval_date = %s,
                            status = %s,
                            data_source = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE anda_number = %s
                        """,
                        (
                            drug_id,
                            anda.get("generic_company"),
                            anda.get("generic_company_ticker"),
                            anda.get("generic_drug_name"),
                            anda.get("dosage_form"),
                            anda.get("strength"),
                            anda.get("filing_date"),
                            anda.get("first_to_file", False),
                            anda.get("paragraph_iv_certification", False),
                            anda.get("tentative_approval_date"),
                            anda.get("final_approval_date"),
                            anda.get("status", "PENDING"),
                            anda.get("data_source", "FDA"),
                            anda.get("anda_number"),
                        ),
                    )
                    updated += 1
                else:
                    # Insert
                    self.db.execute_write(
                        """
                        INSERT INTO generic_applications (
                            anda_number, drug_id, generic_company, generic_company_ticker,
                            generic_drug_name, dosage_form, strength, filing_date,
                            first_to_file, paragraph_iv_certification,
                            tentative_approval_date, final_approval_date, status, data_source
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            anda.get("anda_number"),
                            drug_id,
                            anda.get("generic_company"),
                            anda.get("generic_company_ticker"),
                            anda.get("generic_drug_name"),
                            anda.get("dosage_form"),
                            anda.get("strength"),
                            anda.get("filing_date"),
                            anda.get("first_to_file", False),
                            anda.get("paragraph_iv_certification", False),
                            anda.get("tentative_approval_date"),
                            anda.get("final_approval_date"),
                            anda.get("status", "PENDING"),
                            anda.get("data_source", "FDA"),
                        ),
                    )
                    inserted += 1

            except Exception as e:
                logger.error(f"Error loading ANDA {anda.get('anda_number')}: {e}")

        logger.info(f"ANDAs loaded: {inserted} inserted, {updated} updated")
        return inserted, updated

    def load_calendar_events(
        self, events: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """
        Load patent cliff calendar events.

        Args:
            events: List of calendar event dictionaries.

        Returns:
            Tuple of (inserted_count, updated_count).
        """
        inserted = 0
        updated = 0

        for event in events:
            try:
                # Get drug_id
                drug_result = None
                if event.get("nda_number"):
                    drug_result = self.db.execute_query(
                        "SELECT drug_id FROM drugs WHERE nda_number = %s",
                        (event.get("nda_number"),),
                    )
                elif event.get("drug_id"):
                    drug_result = [{"drug_id": event.get("drug_id")}]

                if not drug_result:
                    logger.warning(f"Drug not found for calendar event")
                    continue

                drug_id = drug_result[0]["drug_id"]

                # Check for existing event (same drug, type, and date)
                existing = self.db.execute_query(
                    """
                    SELECT event_id FROM patent_cliff_calendar
                    WHERE drug_id = %s AND event_type = %s AND event_date = %s
                    """,
                    (drug_id, event.get("event_type"), event.get("event_date")),
                )

                if existing:
                    # Update
                    self.db.execute_write(
                        """
                        UPDATE patent_cliff_calendar SET
                            related_patent_number = %s,
                            related_anda_number = %s,
                            certainty_score = %s,
                            market_opportunity = %s,
                            opportunity_tier = %s,
                            trade_recommendation = %s,
                            recommendation_confidence = %s,
                            notes = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE event_id = %s
                        """,
                        (
                            event.get("related_patent_number"),
                            event.get("related_anda_number"),
                            event.get("certainty_score"),
                            event.get("market_opportunity"),
                            event.get("opportunity_tier"),
                            event.get("trade_recommendation"),
                            event.get("recommendation_confidence"),
                            event.get("notes"),
                            existing[0]["event_id"],
                        ),
                    )
                    updated += 1
                else:
                    # Insert
                    self.db.execute_write(
                        """
                        INSERT INTO patent_cliff_calendar (
                            drug_id, event_type, event_date, related_patent_number,
                            related_anda_number, certainty_score, market_opportunity,
                            opportunity_tier, trade_recommendation, recommendation_confidence,
                            notes
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            drug_id,
                            event.get("event_type"),
                            event.get("event_date"),
                            event.get("related_patent_number"),
                            event.get("related_anda_number"),
                            event.get("certainty_score"),
                            event.get("market_opportunity"),
                            event.get("opportunity_tier"),
                            event.get("trade_recommendation"),
                            event.get("recommendation_confidence"),
                            event.get("notes"),
                        ),
                    )
                    inserted += 1

            except Exception as e:
                logger.error(f"Error loading calendar event: {e}")

        logger.info(f"Calendar events loaded: {inserted} inserted, {updated} updated")
        return inserted, updated

    def log_etl_job(
        self,
        job_name: str,
        job_type: str,
        data_source: str,
        status: str,
        records_processed: int = 0,
        records_inserted: int = 0,
        records_updated: int = 0,
        error_message: Optional[str] = None,
        start_time: Optional[datetime] = None,
    ) -> int:
        """
        Log an ETL job execution.

        Args:
            job_name: Name of the job.
            job_type: Type (EXTRACTION, TRANSFORMATION, LOADING).
            data_source: Source of data.
            status: Job status.
            records_processed: Number of records processed.
            records_inserted: Number of records inserted.
            records_updated: Number of records updated.
            error_message: Error message if failed.
            start_time: Job start time.

        Returns:
            Job ID.
        """
        if start_time is None:
            start_time = datetime.now()

        result = self.db.execute_query(
            """
            INSERT INTO etl_jobs (
                job_name, job_type, data_source, start_time, end_time,
                status, records_processed, records_inserted, records_updated,
                error_message
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING job_id
            """,
            (
                job_name,
                job_type,
                data_source,
                start_time,
                datetime.now() if status != "RUNNING" else None,
                status,
                records_processed,
                records_inserted,
                records_updated,
                error_message,
            ),
        )

        return result[0]["job_id"] if result else 0

    def get_drugs_for_calendar(self) -> List[Dict[str, Any]]:
        """
        Get drugs with their patent and ANDA information for calendar generation.

        Returns:
            List of drug records with related data.
        """
        return self.db.execute_query(
            """
            SELECT
                d.drug_id,
                d.nda_number,
                d.brand_name,
                d.generic_name,
                d.branded_company,
                d.branded_company_ticker,
                d.annual_revenue,
                p.patent_number,
                p.patent_type,
                p.final_expiration_date,
                p.patent_status,
                (SELECT COUNT(*) FROM generic_applications ga
                 WHERE ga.drug_id = d.drug_id AND ga.status = 'APPROVED') as approved_generics,
                (SELECT COUNT(*) FROM litigation l
                 JOIN patents p2 ON l.patent_number = p2.patent_number
                 WHERE p2.drug_id = d.drug_id AND l.outcome = 'ONGOING') as active_litigation
            FROM drugs d
            LEFT JOIN patents p ON d.drug_id = p.drug_id
            WHERE p.final_expiration_date IS NOT NULL
              AND p.patent_status = 'ACTIVE'
            ORDER BY p.final_expiration_date ASC
            """
        )

    def close(self) -> None:
        """Close database connection."""
        self.db.close()
