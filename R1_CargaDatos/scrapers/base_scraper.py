from abc import ABC, abstractmethod
import pandas as pd
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """
    Clase base abstracta para scrapers bibliográficos.
    
    Todos los scrapers deben implementar:
    - scrape(query, max_results): Método principal de scraping
    - _extract_article_data(element): Extraer datos de un artículo
    """
    
    def __init__(self, headless: bool = True, timeout: int = 30):
        """
        Inicializa el scraper.
    
        """
        self.headless = headless
        self.timeout = timeout
        self.driver = None
        self.wait = None
    
    def _init_driver(self):
        """
        Inicializa el driver de Playwright con Chrome.
        
        """
        options = Options()
        
        if self.headless:
            options.add_argument('--headless')
        
        # Opciones para evitar detección
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-gpu')
        
        # User agent
        options.add_argument(
            'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
        
        # Prevenir detección de webdriver
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.implicitly_wait(10)
            self.wait = WebDriverWait(self.driver, self.timeout)
            
            # JavaScript para ocultar webdriver
            self.driver.execute_cdp_cmd(
                'Page.addScriptToEvaluateOnNewDocument',
                {'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'}
            )
            
            logger.info(f"Driver de Playwright inicializado ({'headless' if self.headless else 'visible'})")
            
        except Exception as e:
            logger.error(f"Error inicializando driver: {e}")
            raise
    
    def _close_driver(self):
        """Cierra el driver de Playwright."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Driver cerrado correctamente")
            except Exception as e:
                logger.warning(f"Error cerrando driver: {e}")
    
    @abstractmethod
    def scrape(self, query: str, max_results: int = 100) -> pd.DataFrame:
        """
        Método principal de scraping.
        
        """
        pass
    
    @abstractmethod
    def _extract_article_data(self, element) -> dict:
        """
        Extrae datos de un artículo individual.
    
        """
        pass
    
    def _create_article_dict(self) -> dict:
        """
        Crea diccionario vacío con estructura estándar de artículo.
        
        """
        return {
            'title': '',
            'authors': '',
            'year': '',
            'abstract': '',
            'keywords': '',
            'doi': '',
            'url': '',
            'venue': '',
            'type': '',
            'source': self.__class__.__name__.replace('Scraper', '')
        }
    
    def _safe_extract(self, func, default=''):
        """
        Ejecuta función de extracción con manejo de errores.
        
        """
        try:
            return func()
        except Exception as e:
            logger.debug(f"Error en extracción: {e}")
            return default
    
    def __enter__(self):
        """Context manager: entrada."""
        self._init_driver()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: salida."""
        self._close_driver()