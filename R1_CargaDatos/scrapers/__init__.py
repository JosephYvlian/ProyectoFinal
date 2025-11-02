"""
Scrapers para bases de datos bibliográficas.

Disponibles:
- ACMScraper: ACM Digital Library (vía Playwright)
- IEEEScraper: IEEE Xplore (vía Playwright con login Google)
"""

from .base_scraper import BaseScraper
from .acm_scraper import ACMScraper
from .ieee_scraper import IEEEScraper

__all__ = [
    'BaseScraper',
    'ACMScraper',
    'IEEEScraper'
]