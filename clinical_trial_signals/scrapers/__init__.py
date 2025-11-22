"""Scrapers module for Clinical Trial Signal Detection System."""
from .clinicaltrials import ClinicalTrialsScraper
from .sec_edgar import SECEdgarScraper

__all__ = [
    "ClinicalTrialsScraper",
    "SECEdgarScraper",
]
