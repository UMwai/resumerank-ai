"""Page modules for the Investment Intelligence Dashboard."""

from .home import render_home_page
from .clinical_trials import render_clinical_trials_page
from .patent_cliff import render_patent_cliff_page
from .insider_hiring import render_insider_hiring_page
from .watchlist import render_watchlist_page
from .alerts import render_alerts_page

__all__ = [
    "render_home_page",
    "render_clinical_trials_page",
    "render_patent_cliff_page",
    "render_insider_hiring_page",
    "render_watchlist_page",
    "render_alerts_page",
]
