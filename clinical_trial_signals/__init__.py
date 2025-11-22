"""
Clinical Trial Signal Detection System

An automated intelligence system for detecting early signals of clinical trial
outcomes before public announcements.

Main components:
- scrapers: Data collection from ClinicalTrials.gov and SEC EDGAR
- database: PostgreSQL storage and data models
- scoring: Multi-factor signal scoring model
- alerts: Email digest generation and delivery
- utils: Change detection and utilities
"""

__version__ = "1.0.0"
__author__ = "Clinical Trial Signals Team"
