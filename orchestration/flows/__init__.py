"""
Prefect Flows Module

Alternative orchestration using Prefect instead of Airflow.
Provides the same functionality with a more Pythonic API.

Available Flows:
- clinical_trial_flow: Daily clinical trial signal detection
- patent_ip_flow: Weekly patent cliff analysis (coming soon)
- insider_hiring_flow: Multi-schedule insider/hiring signals (coming soon)

Usage:
    from orchestration.flows import clinical_trial_flow
    result = clinical_trial_flow()
"""

from .clinical_trial_flow import clinical_trial_flow

__all__ = [
    "clinical_trial_flow",
]
