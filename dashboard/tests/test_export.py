"""Tests for the export module."""

import pytest
import pandas as pd
import json
from utils.export import (
    export_to_csv,
    export_to_excel,
    export_to_json,
    format_dataframe_for_export,
)


class TestExportToCSV:
    """Tests for CSV export functionality."""

    def test_basic_export(self):
        """Test basic CSV export."""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL'],
            'score': [0.85, 0.72],
        })
        result = export_to_csv(df)
        assert isinstance(result, bytes)
        assert b'ticker' in result
        assert b'AAPL' in result

    def test_export_with_index(self):
        """Test CSV export with index."""
        df = pd.DataFrame({
            'ticker': ['AAPL'],
            'score': [0.85],
        })
        result = export_to_csv(df, include_index=True)
        assert isinstance(result, bytes)

    def test_empty_dataframe(self):
        """Test CSV export with empty DataFrame."""
        df = pd.DataFrame()
        result = export_to_csv(df)
        assert isinstance(result, bytes)


class TestExportToJSON:
    """Tests for JSON export functionality."""

    def test_basic_export(self):
        """Test basic JSON export."""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL'],
            'score': [0.85, 0.72],
        })
        result = export_to_json(df)
        assert isinstance(result, bytes)
        data = json.loads(result.decode('utf-8'))
        assert len(data) == 2
        assert data[0]['ticker'] == 'AAPL'

    def test_different_orient(self):
        """Test JSON export with different orientation."""
        df = pd.DataFrame({
            'ticker': ['AAPL'],
            'score': [0.85],
        })
        result = export_to_json(df, orient='columns')
        assert isinstance(result, bytes)


class TestExportToExcel:
    """Tests for Excel export functionality."""

    def test_basic_export(self):
        """Test basic Excel export."""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL'],
            'score': [0.85, 0.72],
        })
        result = export_to_excel(df)
        assert isinstance(result, bytes)
        # Excel files start with PK (ZIP header)
        assert result[:2] == b'PK'

    def test_export_with_sheet_name(self):
        """Test Excel export with custom sheet name."""
        df = pd.DataFrame({
            'ticker': ['AAPL'],
            'score': [0.85],
        })
        result = export_to_excel(df, sheet_name='Opportunities')
        assert isinstance(result, bytes)


class TestFormatDataframeForExport:
    """Tests for DataFrame formatting functionality."""

    def test_rename_columns(self):
        """Test column renaming."""
        df = pd.DataFrame({
            'ticker': ['AAPL'],
            'combined_score': [0.85],
        })
        formatted = format_dataframe_for_export(
            df,
            rename_columns={'ticker': 'Symbol', 'combined_score': 'Score'}
        )
        assert 'Symbol' in formatted.columns
        assert 'Score' in formatted.columns

    def test_drop_columns(self):
        """Test column dropping."""
        df = pd.DataFrame({
            'ticker': ['AAPL'],
            'internal_id': [123],
            'score': [0.85],
        })
        formatted = format_dataframe_for_export(
            df,
            drop_columns=['internal_id']
        )
        assert 'internal_id' not in formatted.columns
        assert 'ticker' in formatted.columns

    def test_date_formatting(self):
        """Test date column formatting."""
        df = pd.DataFrame({
            'ticker': ['AAPL'],
            'date': pd.to_datetime(['2024-01-15']),
        })
        formatted = format_dataframe_for_export(
            df,
            date_columns=['date']
        )
        assert formatted.iloc[0]['date'] == '2024-01-15'

    def test_percentage_formatting(self):
        """Test percentage column formatting."""
        df = pd.DataFrame({
            'ticker': ['AAPL'],
            'confidence': [0.85],
        })
        formatted = format_dataframe_for_export(
            df,
            percentage_columns=['confidence']
        )
        assert '85' in formatted.iloc[0]['confidence']

    def test_currency_formatting(self):
        """Test currency column formatting."""
        df = pd.DataFrame({
            'ticker': ['AAPL'],
            'revenue': [1000000.50],
        })
        formatted = format_dataframe_for_export(
            df,
            currency_columns=['revenue']
        )
        assert '$' in formatted.iloc[0]['revenue']
