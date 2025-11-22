"""Initial schema migration

Revision ID: 20241122_000001
Revises:
Create Date: 2024-11-22 00:00:01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '20241122_000001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Companies table
    op.create_table(
        'companies',
        sa.Column('ticker', sa.String(10), primary_key=True),
        sa.Column('company_name', sa.String(200), nullable=False),
        sa.Column('market_cap', sa.BigInteger, nullable=True),
        sa.Column('current_price', sa.Numeric(10, 2), nullable=True),
        sa.Column('sector', sa.String(100), server_default='Biotechnology'),
        sa.Column('cik', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.current_timestamp()),
    )

    # Trials table
    op.create_table(
        'trials',
        sa.Column('trial_id', sa.String(20), primary_key=True),
        sa.Column('company_ticker', sa.String(10), sa.ForeignKey('companies.ticker'), nullable=True),
        sa.Column('drug_name', sa.String(200), nullable=True),
        sa.Column('indication', sa.String(500), nullable=True),
        sa.Column('phase', sa.String(10), nullable=True),
        sa.Column('enrollment_target', sa.Integer, nullable=True),
        sa.Column('enrollment_current', sa.Integer, nullable=True),
        sa.Column('start_date', sa.Date, nullable=True),
        sa.Column('expected_completion', sa.Date, nullable=True),
        sa.Column('primary_completion_date', sa.Date, nullable=True),
        sa.Column('primary_endpoint', sa.Text, nullable=True),
        sa.Column('status', sa.String(50), nullable=True),
        sa.Column('sponsor', sa.String(300), nullable=True),
        sa.Column('study_type', sa.String(50), nullable=True),
        sa.Column('last_updated', sa.DateTime, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),
        sa.Column('raw_data', postgresql.JSONB, nullable=True),
    )

    # Trial signals table
    op.create_table(
        'trial_signals',
        sa.Column('signal_id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('trial_id', sa.String(20), sa.ForeignKey('trials.trial_id'), nullable=True),
        sa.Column('signal_type', sa.String(100), nullable=False),
        sa.Column('signal_value', sa.Text, nullable=True),
        sa.Column('signal_weight', sa.Integer, nullable=True),
        sa.Column('detected_date', sa.DateTime, server_default=sa.func.current_timestamp()),
        sa.Column('source', sa.String(100), nullable=True),
        sa.Column('source_url', sa.Text, nullable=True),
        sa.Column('raw_data', postgresql.JSONB, nullable=True),
        sa.Column('processed', sa.Boolean, server_default='false'),
        sa.CheckConstraint('signal_weight >= -5 AND signal_weight <= 5', name='ck_signal_weight'),
    )

    # Trial scores table
    op.create_table(
        'trial_scores',
        sa.Column('score_id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('trial_id', sa.String(20), sa.ForeignKey('trials.trial_id'), nullable=True),
        sa.Column('score_date', sa.Date, server_default=sa.func.current_date()),
        sa.Column('composite_score', sa.Numeric(4, 2), nullable=True),
        sa.Column('confidence', sa.Numeric(3, 2), nullable=True),
        sa.Column('recommendation', sa.String(20), nullable=True),
        sa.Column('contributing_signals', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),
        sa.UniqueConstraint('trial_id', 'score_date', name='uq_trial_score_date'),
        sa.CheckConstraint('composite_score >= 0 AND composite_score <= 10', name='ck_composite_score'),
        sa.CheckConstraint('confidence >= 0 AND confidence <= 1', name='ck_confidence'),
    )

    # SEC filings table
    op.create_table(
        'sec_filings',
        sa.Column('filing_id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('company_ticker', sa.String(10), sa.ForeignKey('companies.ticker'), nullable=True),
        sa.Column('filing_type', sa.String(20), nullable=True),
        sa.Column('filing_date', sa.Date, nullable=True),
        sa.Column('accession_number', sa.String(30), unique=True, nullable=True),
        sa.Column('filing_url', sa.Text, nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('raw_content', sa.Text, nullable=True),
        sa.Column('processed', sa.Boolean, server_default='false'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),
    )

    # Trial history table
    op.create_table(
        'trial_history',
        sa.Column('history_id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('trial_id', sa.String(20), sa.ForeignKey('trials.trial_id'), nullable=True),
        sa.Column('snapshot_date', sa.DateTime, server_default=sa.func.current_timestamp()),
        sa.Column('enrollment_current', sa.Integer, nullable=True),
        sa.Column('status', sa.String(50), nullable=True),
        sa.Column('expected_completion', sa.Date, nullable=True),
        sa.Column('primary_endpoint', sa.Text, nullable=True),
        sa.Column('raw_data', postgresql.JSONB, nullable=True),
    )

    # Email digests table
    op.create_table(
        'email_digests',
        sa.Column('digest_id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('sent_at', sa.DateTime, server_default=sa.func.current_timestamp()),
        sa.Column('recipients', postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column('subject', sa.String(300), nullable=True),
        sa.Column('signals_count', sa.Integer, nullable=True),
        sa.Column('content_hash', sa.String(64), nullable=True),
    )

    # USPTO patents table (new for Phase 1)
    op.create_table(
        'patents',
        sa.Column('patent_id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('company_ticker', sa.String(10), sa.ForeignKey('companies.ticker'), nullable=True),
        sa.Column('patent_number', sa.String(50), unique=True, nullable=True),
        sa.Column('application_number', sa.String(50), nullable=True),
        sa.Column('title', sa.Text, nullable=True),
        sa.Column('abstract', sa.Text, nullable=True),
        sa.Column('filing_date', sa.Date, nullable=True),
        sa.Column('publication_date', sa.Date, nullable=True),
        sa.Column('grant_date', sa.Date, nullable=True),
        sa.Column('inventors', postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column('assignee', sa.String(300), nullable=True),
        sa.Column('patent_url', sa.Text, nullable=True),
        sa.Column('processed', sa.Boolean, server_default='false'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),
    )

    # Preprints table (new for Phase 1)
    op.create_table(
        'preprints',
        sa.Column('preprint_id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('trial_id', sa.String(20), sa.ForeignKey('trials.trial_id'), nullable=True),
        sa.Column('company_ticker', sa.String(10), sa.ForeignKey('companies.ticker'), nullable=True),
        sa.Column('source', sa.String(50), nullable=True),  # pubmed, medrxiv, biorxiv
        sa.Column('external_id', sa.String(100), unique=True, nullable=True),  # PMID or DOI
        sa.Column('title', sa.Text, nullable=True),
        sa.Column('abstract', sa.Text, nullable=True),
        sa.Column('authors', postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column('publication_date', sa.Date, nullable=True),
        sa.Column('journal', sa.String(300), nullable=True),
        sa.Column('url', sa.Text, nullable=True),
        sa.Column('keywords', postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column('processed', sa.Boolean, server_default='false'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),
    )

    # Create indexes
    op.create_index('idx_trials_ticker', 'trials', ['company_ticker'])
    op.create_index('idx_trials_status', 'trials', ['status'])
    op.create_index('idx_trials_phase', 'trials', ['phase'])
    op.create_index('idx_signals_trial', 'trial_signals', ['trial_id'])
    op.create_index('idx_signals_date', 'trial_signals', ['detected_date'])
    op.create_index('idx_signals_type', 'trial_signals', ['signal_type'])
    op.create_index('idx_scores_trial', 'trial_scores', ['trial_id'])
    op.create_index('idx_scores_date', 'trial_scores', ['score_date'])
    op.create_index('idx_sec_filings_ticker', 'sec_filings', ['company_ticker'])
    op.create_index('idx_sec_filings_date', 'sec_filings', ['filing_date'])
    op.create_index('idx_trial_history_trial', 'trial_history', ['trial_id'])
    op.create_index('idx_trial_history_date', 'trial_history', ['snapshot_date'])
    op.create_index('idx_patents_ticker', 'patents', ['company_ticker'])
    op.create_index('idx_patents_filing_date', 'patents', ['filing_date'])
    op.create_index('idx_preprints_trial', 'preprints', ['trial_id'])
    op.create_index('idx_preprints_date', 'preprints', ['publication_date'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_preprints_date')
    op.drop_index('idx_preprints_trial')
    op.drop_index('idx_patents_filing_date')
    op.drop_index('idx_patents_ticker')
    op.drop_index('idx_trial_history_date')
    op.drop_index('idx_trial_history_trial')
    op.drop_index('idx_sec_filings_date')
    op.drop_index('idx_sec_filings_ticker')
    op.drop_index('idx_scores_date')
    op.drop_index('idx_scores_trial')
    op.drop_index('idx_signals_type')
    op.drop_index('idx_signals_date')
    op.drop_index('idx_signals_trial')
    op.drop_index('idx_trials_phase')
    op.drop_index('idx_trials_status')
    op.drop_index('idx_trials_ticker')

    # Drop tables in reverse order of creation
    op.drop_table('preprints')
    op.drop_table('patents')
    op.drop_table('email_digests')
    op.drop_table('trial_history')
    op.drop_table('sec_filings')
    op.drop_table('trial_scores')
    op.drop_table('trial_signals')
    op.drop_table('trials')
    op.drop_table('companies')
