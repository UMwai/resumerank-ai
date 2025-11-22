"""Initial database schema for Patent Intelligence System

Revision ID: 001
Revises: None
Create Date: 2024-11-22

This migration creates the initial database schema including:
- Core tables: drugs, patents, generic_applications, litigation
- Calendar and tracking tables
- Supporting tables for companies, revenue history, etc.
- Indexes, views, and functions
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema."""

    # Enable extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm"')

    # Create drugs table
    op.create_table(
        'drugs',
        sa.Column('drug_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('nda_number', sa.String(20), unique=True),
        sa.Column('brand_name', sa.String(200), nullable=False),
        sa.Column('generic_name', sa.String(200)),
        sa.Column('active_ingredient', sa.String(500)),
        sa.Column('branded_company', sa.String(200)),
        sa.Column('branded_company_ticker', sa.String(10)),
        sa.Column('therapeutic_area', sa.String(200)),
        sa.Column('dosage_form', sa.String(100)),
        sa.Column('route_of_administration', sa.String(100)),
        sa.Column('annual_revenue', sa.BigInteger()),
        sa.Column('revenue_year', sa.Integer()),
        sa.Column('fda_approval_date', sa.Date()),
        sa.Column('market_status', sa.String(50), server_default='ACTIVE'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
    )

    # Create drugs indexes
    op.create_index('idx_drugs_brand_name', 'drugs', ['brand_name'])
    op.create_index('idx_drugs_generic_name', 'drugs', ['generic_name'])
    op.create_index('idx_drugs_company_ticker', 'drugs', ['branded_company_ticker'])
    op.create_index('idx_drugs_therapeutic_area', 'drugs', ['therapeutic_area'])
    op.create_index('idx_drugs_annual_revenue', 'drugs', [sa.text('annual_revenue DESC')])

    # Create patents table
    op.create_table(
        'patents',
        sa.Column('patent_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('patent_number', sa.String(20), unique=True, nullable=False),
        sa.Column('drug_id', sa.Integer(), sa.ForeignKey('drugs.drug_id', ondelete='CASCADE')),
        sa.Column('patent_type', sa.String(50)),
        sa.Column('patent_use_code', sa.String(50)),
        sa.Column('patent_claims', sa.Text()),
        sa.Column('filing_date', sa.Date()),
        sa.Column('grant_date', sa.Date()),
        sa.Column('base_expiration_date', sa.Date(), nullable=False),
        sa.Column('pta_days', sa.Integer(), server_default='0'),
        sa.Column('pte_days', sa.Integer(), server_default='0'),
        sa.Column('adjusted_expiration_date', sa.Date()),
        sa.Column('pediatric_exclusivity_date', sa.Date()),
        sa.Column('final_expiration_date', sa.Date()),
        sa.Column('exclusivity_code', sa.String(50)),
        sa.Column('exclusivity_expiration', sa.Date()),
        sa.Column('patent_status', sa.String(50), server_default='ACTIVE'),
        sa.Column('strength_score', sa.Integer(), sa.CheckConstraint('strength_score >= 1 AND strength_score <= 10')),
        sa.Column('data_source', sa.String(50), server_default='ORANGE_BOOK'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
    )

    # Create patents indexes
    op.create_index('idx_patents_drug_id', 'patents', ['drug_id'])
    op.create_index('idx_patents_expiration', 'patents', ['final_expiration_date'])
    op.create_index('idx_patents_status', 'patents', ['patent_status'])
    op.create_index('idx_patents_type', 'patents', ['patent_type'])

    # Create generic_applications table
    op.create_table(
        'generic_applications',
        sa.Column('anda_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('anda_number', sa.String(20), unique=True, nullable=False),
        sa.Column('drug_id', sa.Integer(), sa.ForeignKey('drugs.drug_id', ondelete='CASCADE')),
        sa.Column('generic_company', sa.String(200), nullable=False),
        sa.Column('generic_company_ticker', sa.String(10)),
        sa.Column('generic_drug_name', sa.String(200)),
        sa.Column('dosage_form', sa.String(100)),
        sa.Column('strength', sa.String(100)),
        sa.Column('filing_date', sa.Date()),
        sa.Column('first_to_file', sa.Boolean(), server_default='false'),
        sa.Column('paragraph_iv_certification', sa.Boolean(), server_default='false'),
        sa.Column('tentative_approval_date', sa.Date()),
        sa.Column('final_approval_date', sa.Date()),
        sa.Column('status', sa.String(50), server_default='PENDING'),
        sa.Column('approval_notes', sa.Text()),
        sa.Column('data_source', sa.String(50), server_default='FDA'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
    )

    # Create generic_applications indexes
    op.create_index('idx_anda_drug_id', 'generic_applications', ['drug_id'])
    op.create_index('idx_anda_company', 'generic_applications', ['generic_company'])
    op.create_index('idx_anda_status', 'generic_applications', ['status'])
    op.create_index('idx_anda_approval_date', 'generic_applications', ['final_approval_date'])
    op.create_index('idx_anda_first_to_file', 'generic_applications', ['first_to_file'])

    # Create litigation table
    op.create_table(
        'litigation',
        sa.Column('litigation_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('case_id', sa.String(50), unique=True, nullable=False),
        sa.Column('case_name', sa.String(500)),
        sa.Column('patent_number', sa.String(20), sa.ForeignKey('patents.patent_number', ondelete='CASCADE')),
        sa.Column('anda_number', sa.String(20), sa.ForeignKey('generic_applications.anda_number', ondelete='SET NULL')),
        sa.Column('plaintiff', sa.String(200)),
        sa.Column('defendant', sa.String(200)),
        sa.Column('court', sa.String(200)),
        sa.Column('jurisdiction', sa.String(100)),
        sa.Column('case_type', sa.String(50)),
        sa.Column('filing_date', sa.Date()),
        sa.Column('trial_date', sa.Date()),
        sa.Column('decision_date', sa.Date()),
        sa.Column('outcome', sa.String(50)),
        sa.Column('settlement_terms', sa.Text()),
        sa.Column('damages_awarded', sa.BigInteger()),
        sa.Column('case_summary', sa.Text()),
        sa.Column('pacer_link', sa.String(500)),
        sa.Column('data_source', sa.String(50), server_default='MANUAL'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
    )

    # Create litigation indexes
    op.create_index('idx_litigation_patent', 'litigation', ['patent_number'])
    op.create_index('idx_litigation_anda', 'litigation', ['anda_number'])
    op.create_index('idx_litigation_outcome', 'litigation', ['outcome'])
    op.create_index('idx_litigation_decision_date', 'litigation', ['decision_date'])

    # Create patent_cliff_calendar table
    op.create_table(
        'patent_cliff_calendar',
        sa.Column('event_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('drug_id', sa.Integer(), sa.ForeignKey('drugs.drug_id', ondelete='CASCADE')),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('event_date', sa.Date(), nullable=False),
        sa.Column('related_patent_number', sa.String(20)),
        sa.Column('related_anda_number', sa.String(20)),
        sa.Column('certainty_score', sa.Numeric(5, 2)),
        sa.Column('market_opportunity', sa.BigInteger()),
        sa.Column('opportunity_tier', sa.String(50)),
        sa.Column('trade_recommendation', sa.Text()),
        sa.Column('recommendation_confidence', sa.String(20)),
        sa.Column('notes', sa.Text()),
        sa.Column('is_processed', sa.Boolean(), server_default='false'),
        sa.Column('alert_sent', sa.Boolean(), server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
    )

    # Create calendar indexes
    op.create_index('idx_calendar_drug_id', 'patent_cliff_calendar', ['drug_id'])
    op.create_index('idx_calendar_event_date', 'patent_cliff_calendar', ['event_date'])
    op.create_index('idx_calendar_event_type', 'patent_cliff_calendar', ['event_type'])
    op.create_index('idx_calendar_certainty', 'patent_cliff_calendar', [sa.text('certainty_score DESC')])
    op.create_index('idx_calendar_opportunity', 'patent_cliff_calendar', [sa.text('market_opportunity DESC')])

    # Create drug_revenue_history table
    op.create_table(
        'drug_revenue_history',
        sa.Column('revenue_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('drug_id', sa.Integer(), sa.ForeignKey('drugs.drug_id', ondelete='CASCADE')),
        sa.Column('fiscal_year', sa.Integer(), nullable=False),
        sa.Column('annual_revenue', sa.BigInteger()),
        sa.Column('revenue_source', sa.String(100)),
        sa.Column('notes', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
        sa.UniqueConstraint('drug_id', 'fiscal_year', name='uq_drug_revenue_year'),
    )

    op.create_index('idx_revenue_drug_id', 'drug_revenue_history', ['drug_id'])
    op.create_index('idx_revenue_year', 'drug_revenue_history', [sa.text('fiscal_year DESC')])

    # Create companies table
    op.create_table(
        'companies',
        sa.Column('company_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('company_name', sa.String(200), nullable=False, unique=True),
        sa.Column('ticker_symbol', sa.String(10)),
        sa.Column('company_type', sa.String(50)),
        sa.Column('headquarters_country', sa.String(100)),
        sa.Column('market_cap', sa.BigInteger()),
        sa.Column('is_public', sa.Boolean(), server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
    )

    op.create_index('idx_companies_ticker', 'companies', ['ticker_symbol'])
    op.create_index('idx_companies_type', 'companies', ['company_type'])

    # Create exclusivity_periods table
    op.create_table(
        'exclusivity_periods',
        sa.Column('exclusivity_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('drug_id', sa.Integer(), sa.ForeignKey('drugs.drug_id', ondelete='CASCADE')),
        sa.Column('exclusivity_type', sa.String(50), nullable=False),
        sa.Column('start_date', sa.Date()),
        sa.Column('end_date', sa.Date(), nullable=False),
        sa.Column('notes', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
    )

    op.create_index('idx_exclusivity_drug_id', 'exclusivity_periods', ['drug_id'])
    op.create_index('idx_exclusivity_end_date', 'exclusivity_periods', ['end_date'])

    # Create etl_jobs table
    op.create_table(
        'etl_jobs',
        sa.Column('job_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('job_name', sa.String(100), nullable=False),
        sa.Column('job_type', sa.String(50)),
        sa.Column('data_source', sa.String(50)),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_time', sa.DateTime(timezone=True)),
        sa.Column('status', sa.String(50), server_default='RUNNING'),
        sa.Column('records_processed', sa.Integer(), server_default='0'),
        sa.Column('records_inserted', sa.Integer(), server_default='0'),
        sa.Column('records_updated', sa.Integer(), server_default='0'),
        sa.Column('error_message', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
    )

    op.create_index('idx_etl_jobs_name', 'etl_jobs', ['job_name'])
    op.create_index('idx_etl_jobs_status', 'etl_jobs', ['status'])
    op.create_index('idx_etl_jobs_start_time', 'etl_jobs', [sa.text('start_time DESC')])

    # Create alert_history table
    op.create_table(
        'alert_history',
        sa.Column('alert_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('event_id', sa.Integer(), sa.ForeignKey('patent_cliff_calendar.event_id', ondelete='SET NULL')),
        sa.Column('alert_type', sa.String(50), nullable=False),
        sa.Column('recipient', sa.String(200)),
        sa.Column('subject', sa.String(500)),
        sa.Column('body', sa.Text()),
        sa.Column('sent_at', sa.DateTime(timezone=True)),
        sa.Column('status', sa.String(50), server_default='PENDING'),
        sa.Column('error_message', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.current_timestamp()),
    )

    op.create_index('idx_alerts_event_id', 'alert_history', ['event_id'])
    op.create_index('idx_alerts_sent_at', 'alert_history', [sa.text('sent_at DESC')])
    op.create_index('idx_alerts_status', 'alert_history', ['status'])

    # Create update timestamp function
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Create triggers for auto-updating timestamps
    for table in ['drugs', 'patents', 'generic_applications', 'litigation',
                  'patent_cliff_calendar', 'companies']:
        op.execute(f"""
            CREATE TRIGGER update_{table}_updated_at
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """)


def downgrade() -> None:
    """Drop all tables and extensions."""

    # Drop triggers
    for table in ['drugs', 'patents', 'generic_applications', 'litigation',
                  'patent_cliff_calendar', 'companies']:
        op.execute(f"DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table}")

    # Drop function
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")

    # Drop tables in reverse order of dependencies
    op.drop_table('alert_history')
    op.drop_table('etl_jobs')
    op.drop_table('exclusivity_periods')
    op.drop_table('drug_revenue_history')
    op.drop_table('patent_cliff_calendar')
    op.drop_table('litigation')
    op.drop_table('generic_applications')
    op.drop_table('patents')
    op.drop_table('companies')
    op.drop_table('drugs')

    # Drop extensions
    op.execute('DROP EXTENSION IF EXISTS "pg_trgm"')
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')
