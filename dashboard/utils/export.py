"""
Data Export Module for Investment Intelligence Dashboard.

Provides functionality to export data in various formats:
- CSV (Comma-separated values)
- Excel (.xlsx)
- JSON

Features:
- Streamlit download buttons
- Formatting and styling for Excel exports
- Batch export capabilities
"""

import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def export_to_csv(
    df: pd.DataFrame,
    filename: Optional[str] = None,
    include_index: bool = False,
) -> bytes:
    """
    Export DataFrame to CSV format.

    Args:
        df: DataFrame to export
        filename: Optional filename (without extension)
        include_index: Whether to include DataFrame index

    Returns:
        CSV data as bytes
    """
    return df.to_csv(index=include_index).encode('utf-8')


def export_to_excel(
    df: pd.DataFrame,
    sheet_name: str = "Data",
    include_index: bool = False,
    format_numbers: bool = True,
) -> bytes:
    """
    Export DataFrame to Excel format with optional formatting.

    Args:
        df: DataFrame to export
        sheet_name: Name for the Excel sheet
        include_index: Whether to include DataFrame index
        format_numbers: Whether to apply number formatting

    Returns:
        Excel data as bytes
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=include_index,
        )

        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # Auto-adjust column widths
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            ) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)

        # Apply number formatting if requested
        if format_numbers:
            from openpyxl.styles import numbers
            for row in worksheet.iter_rows(min_row=2, max_row=len(df) + 1):
                for cell in row:
                    if isinstance(cell.value, float):
                        if 0 <= cell.value <= 1:
                            cell.number_format = '0.00%' if 'confidence' in str(df.columns[cell.column - 1]).lower() else '0.00'
                        else:
                            cell.number_format = '#,##0.00'

    return output.getvalue()


def export_to_json(
    df: pd.DataFrame,
    orient: str = "records",
    date_format: str = "iso",
) -> bytes:
    """
    Export DataFrame to JSON format.

    Args:
        df: DataFrame to export
        orient: JSON orientation ('records', 'index', 'columns', etc.)
        date_format: Date format ('iso', 'epoch')

    Returns:
        JSON data as bytes
    """
    return df.to_json(orient=orient, date_format=date_format).encode('utf-8')


def create_download_button(
    df: pd.DataFrame,
    format_type: str = "csv",
    filename: str = "export",
    button_label: str = "Download",
    key: Optional[str] = None,
) -> bool:
    """
    Create a Streamlit download button for the DataFrame.

    Args:
        df: DataFrame to export
        format_type: Export format ('csv', 'excel', 'json')
        filename: Base filename (without extension)
        button_label: Label for the download button
        key: Unique key for the button

    Returns:
        True if button was clicked
    """
    if df.empty:
        st.warning("No data available for export.")
        return False

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename}_{timestamp}"

    if format_type.lower() == "csv":
        data = export_to_csv(df)
        mime_type = "text/csv"
        extension = "csv"
    elif format_type.lower() in ["excel", "xlsx"]:
        data = export_to_excel(df)
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        extension = "xlsx"
    elif format_type.lower() == "json":
        data = export_to_json(df)
        mime_type = "application/json"
        extension = "json"
    else:
        st.error(f"Unsupported export format: {format_type}")
        return False

    return st.download_button(
        label=button_label,
        data=data,
        file_name=f"{full_filename}.{extension}",
        mime=mime_type,
        key=key,
    )


def create_export_buttons(
    df: pd.DataFrame,
    filename: str = "data",
    key_prefix: str = "export",
    show_csv: bool = True,
    show_excel: bool = True,
    show_json: bool = False,
) -> None:
    """
    Create a row of export buttons for different formats.

    Args:
        df: DataFrame to export
        filename: Base filename for exports
        key_prefix: Prefix for button keys
        show_csv: Show CSV download button
        show_excel: Show Excel download button
        show_json: Show JSON download button
    """
    if df.empty:
        st.info("No data available for export.")
        return

    # Calculate number of buttons to show
    button_count = sum([show_csv, show_excel, show_json])
    if button_count == 0:
        return

    cols = st.columns(button_count)
    col_idx = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if show_csv:
        with cols[col_idx]:
            st.download_button(
                label="Download CSV",
                data=export_to_csv(df),
                file_name=f"{filename}_{timestamp}.csv",
                mime="text/csv",
                key=f"{key_prefix}_csv",
                use_container_width=True,
            )
        col_idx += 1

    if show_excel:
        with cols[col_idx]:
            st.download_button(
                label="Download Excel",
                data=export_to_excel(df),
                file_name=f"{filename}_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{key_prefix}_excel",
                use_container_width=True,
            )
        col_idx += 1

    if show_json:
        with cols[col_idx]:
            st.download_button(
                label="Download JSON",
                data=export_to_json(df),
                file_name=f"{filename}_{timestamp}.json",
                mime="application/json",
                key=f"{key_prefix}_json",
                use_container_width=True,
            )


def export_multiple_sheets(
    dataframes: Dict[str, pd.DataFrame],
    filename: str = "export",
) -> bytes:
    """
    Export multiple DataFrames to a single Excel file with multiple sheets.

    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        filename: Base filename (without extension)

    Returns:
        Excel data as bytes
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dataframes.items():
            if not df.empty:
                # Clean sheet name (max 31 chars, no special chars)
                clean_name = sheet_name[:31].replace('/', '-').replace('\\', '-')
                df.to_excel(writer, sheet_name=clean_name, index=False)

                # Auto-adjust column widths
                worksheet = writer.sheets[clean_name]
                for idx, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).map(len).max(),
                        len(str(col))
                    ) + 2
                    worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)

    return output.getvalue()


def create_multi_sheet_download(
    dataframes: Dict[str, pd.DataFrame],
    filename: str = "report",
    button_label: str = "Download Full Report",
    key: Optional[str] = None,
) -> bool:
    """
    Create a download button for a multi-sheet Excel export.

    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        filename: Base filename
        button_label: Label for download button
        key: Unique button key

    Returns:
        True if button was clicked
    """
    # Filter out empty DataFrames
    valid_dfs = {k: v for k, v in dataframes.items() if not v.empty}

    if not valid_dfs:
        st.warning("No data available for export.")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return st.download_button(
        label=button_label,
        data=export_multiple_sheets(valid_dfs, filename),
        file_name=f"{filename}_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,
    )


def format_dataframe_for_export(
    df: pd.DataFrame,
    rename_columns: Optional[Dict[str, str]] = None,
    drop_columns: Optional[List[str]] = None,
    date_columns: Optional[List[str]] = None,
    percentage_columns: Optional[List[str]] = None,
    currency_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Format a DataFrame for export with human-readable column names and formatting.

    Args:
        df: DataFrame to format
        rename_columns: Dictionary mapping old names to new names
        drop_columns: List of columns to drop
        date_columns: List of columns to format as dates
        percentage_columns: List of columns to format as percentages
        currency_columns: List of columns to format as currency

    Returns:
        Formatted DataFrame
    """
    export_df = df.copy()

    # Drop specified columns
    if drop_columns:
        export_df = export_df.drop(columns=[c for c in drop_columns if c in export_df.columns])

    # Format date columns
    if date_columns:
        for col in date_columns:
            if col in export_df.columns:
                export_df[col] = pd.to_datetime(export_df[col]).dt.strftime('%Y-%m-%d')

    # Format percentage columns
    if percentage_columns:
        for col in percentage_columns:
            if col in export_df.columns:
                export_df[col] = export_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")

    # Format currency columns
    if currency_columns:
        for col in currency_columns:
            if col in export_df.columns:
                export_df[col] = export_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")

    # Rename columns
    if rename_columns:
        export_df = export_df.rename(columns=rename_columns)

    return export_df


def export_opportunities(
    df: pd.DataFrame,
    filename: str = "opportunities",
) -> None:
    """
    Export opportunities data with proper formatting.

    Args:
        df: DataFrame with opportunity data
        filename: Base filename
    """
    # Format for export
    export_df = format_dataframe_for_export(
        df,
        rename_columns={
            'ticker': 'Ticker',
            'company_name': 'Company',
            'combined_score': 'Combined Score',
            'confidence': 'Confidence',
            'recommendation': 'Recommendation',
            'signal_count': 'Signal Count',
            'clinical_score': 'Clinical Score',
            'patent_score': 'Patent Score',
            'insider_score': 'Insider Score',
        },
        percentage_columns=['Confidence'],
    )

    create_export_buttons(
        export_df,
        filename=filename,
        key_prefix=f"{filename}_export",
    )


def export_signals(
    df: pd.DataFrame,
    signal_type: str = "signals",
    filename: Optional[str] = None,
) -> None:
    """
    Export signals data with proper formatting.

    Args:
        df: DataFrame with signal data
        signal_type: Type of signals for filename
        filename: Optional custom filename
    """
    if filename is None:
        filename = f"{signal_type}_export"

    create_export_buttons(
        df,
        filename=filename,
        key_prefix=f"{signal_type}_export",
    )
