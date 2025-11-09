"""
ACM Scraper con Soluciones Anti-CAPTCHA
========================================

Soluciones implementadas:
1. User-Agent realista
2. Delays aleatorios (comportamiento humano)
3. Modo NO headless (CAPTCHA manual)
4. Stealth plugin para Playwright
5. Cookies persistentes
"""

import time
import random
import pandas as pd
import re
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ACMScraper:
    """
    Scraper para ACM Digital Library con anti-detección.
    """
    
    def __init__(self, email: str = "josephy.garciac@uqvirtual.edu.co"):
        self.email = email
        self.base_url = "https://library.uniquindio.edu.co/databases"
        self.cookies_file = Path("data/cache/acm_cookies.json")
        self.cookies_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _random_delay(self, min_sec: float = 1.5, max_sec: float = 3.5):
        """Delay aleatorio para simular comportamiento humano"""
        time.sleep(random.uniform(min_sec, max_sec))
    
    def _save_cookies(self, context):
        """Guarda cookies para reutilizar sesión"""
        try:
            cookies = context.cookies()
            with open(self.cookies_file, 'w') as f:
                json.dump(cookies, f)
            logger.info(f"  → Cookies guardadas en: {self.cookies_file}")
        except Exception as e:
            logger.warning(f"  → No se pudieron guardar cookies: {e}")
    
    def _load_cookies(self, context):
        """Carga cookies guardadas previamente"""
        try:
            if self.cookies_file.exists():
                with open(self.cookies_file, 'r') as f:
                    cookies = json.load(f)
                context.add_cookies(cookies)
                logger.info("  → Cookies cargadas exitosamente")
                return True
        except Exception as e:
            logger.warning(f"  → No se pudieron cargar cookies: {e}")
        return False
    
    def scrape(self, query: str, max_results: int = 100, manual_captcha: bool = True) -> pd.DataFrame:
        """
        Scrapea artículos de ACM con anti-CAPTCHA.
        
        Args:
            query: Consulta de búsqueda
            max_results: Máximo de artículos a obtener
            manual_captcha: Si True, pausa para resolver CAPTCHA manualmente
        """
        logger.info(f"Iniciando scraping de ACM: '{query}'")
        logger.info(f"Acceso institucional: {self.email}")
        
        from playwright.sync_api import sync_playwright
        
        articles = []
        
        try:
            with sync_playwright() as p:
                start_time = time.time()
                
                # ============================================
                # CONFIGURACIÓN ANTI-DETECCIÓN
                # ============================================
                
                # Lanzar navegador con user-agent realista
                browser = p.chromium.launch(
                    headless=False,  # NO headless para ver CAPTCHA
                    args=[
                        '--disable-blink-features=AutomationControlled',  # Ocultar automatización
                        '--disable-dev-shm-usage',
                        '--no-sandbox'
                    ]
                )
                
                # Crear contexto con user-agent de navegador real
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080},
                    locale='es-CO',
                    timezone_id='America/Bogota'
                )
                
                # Cargar cookies si existen
                cookies_loaded = self._load_cookies(context)
                
                page = context.new_page()
                
                # Ocultar propiedades de WebDriver
                page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                    
                    // Override the Chrome property
                    window.chrome = {
                        runtime: {}
                    };
                    
                    // Override permissions
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                    );
                """)
                
                # ============================================
                # PASO 1: Acceder a biblioteca UQ
                # ============================================
                logger.info("  → Accediendo a biblioteca Universidad del Quindío...")
                page.goto(self.base_url)
                self._random_delay(2, 4)
                page.wait_for_load_state("domcontentloaded")
                
                # ============================================
                # PASO 2: Click en Facultad de Ingeniería
                # ============================================
                fac_ingenieria_selector = "div[data-content-listing-item='fac-ingenier-a']"
                page.click(fac_ingenieria_selector)
                self._random_delay(1.5, 3)
                page.wait_for_load_state("domcontentloaded")
                
                # ============================================
                # PASO 3: Click en ACM Digital Library
                # ============================================
                logger.info("  → Accediendo a ACM Digital Library...")
                elements = page.locator("//a[contains(@href, 'dl.acm.org')]//span[contains(text(), 'ACM Digital Library')]")
                count = elements.count()
                
                clicked = False
                for i in range(count):
                    if elements.nth(i).is_visible():
                        elements.nth(i).click()
                        self._random_delay(3, 5)  # Espera más larga
                        page.wait_for_load_state("domcontentloaded")
                        logger.info("  → Acceso a ACM exitoso")
                        clicked = True
                        break
                
                if not clicked:
                    logger.error("No se pudo acceder a ACM Digital Library")
                    browser.close()
                    return pd.DataFrame()
                
                # ============================================
                # DETECCIÓN Y RESOLUCIÓN DE CAPTCHA
                # ============================================
                
                # Esperar un momento para ver si aparece CAPTCHA
                self._random_delay(2, 3)
                
                # Verificar si hay CAPTCHA
                captcha_selectors = [
                    "iframe[src*='captcha']",
                    "iframe[src*='recaptcha']",
                    ".g-recaptcha",
                    "#captcha-box",
                    "div[class*='captcha']"
                ]
                
                captcha_found = False
                for selector in captcha_selectors:
                    try:
                        if page.query_selector(selector):
                            captcha_found = True
                            logger.warning("\n" + "="*70)
                            logger.warning("CAPTCHA DETECTADO")
                            logger.warning("="*70)
                            break
                    except:
                        pass
                
                if captcha_found and manual_captcha:
                    logger.warning("\n🔴 ACCIÓN REQUERIDA:")
                    logger.warning("   1. Resuelve el CAPTCHA en el navegador abierto")
                    logger.warning("   2. Espera a que la página cargue completamente")
                    logger.warning("   3. Presiona ENTER aquí para continuar...")
                    logger.warning("="*70 + "\n")
                    input(">>> Presiona ENTER después de resolver el CAPTCHA: ")
                    logger.info("  → Continuando scraping...")
                    self._random_delay(1, 2)
                elif captcha_found:
                    logger.error("CAPTCHA detectado. Configure manual_captcha=True")
                    browser.close()
                    return pd.DataFrame()
                
                # Guardar cookies después de pasar CAPTCHA
                self._save_cookies(context)
                
                # ============================================
                # PASO 4: Realizar búsqueda
                # ============================================
                logger.info(f"  → Buscando: '{query}'...")
                search_selector = "input[name='AllField']"
                
                try:
                    page.wait_for_selector(search_selector, timeout=60000)
                    self._random_delay(1, 2)
                    
                    # Escribir query letra por letra (más humano)
                    page.click(search_selector)
                    for char in query:
                        page.keyboard.type(char)
                        time.sleep(random.uniform(0.05, 0.15))
                    
                    self._random_delay(0.5, 1)
                    page.press(search_selector, "Enter")
                    page.wait_for_selector(".search__item", timeout=60000)
                    
                except Exception as e:
                    logger.error(f"  → Error en búsqueda: {e}")
                    browser.close()
                    return pd.DataFrame()
                
                # ============================================
                # PASO 5: Cambiar a 50 artículos por página
                # ============================================
                try:
                    link_50_selector = "a[href*='pageSize=50']"
                    page.wait_for_selector(link_50_selector, timeout=10000)
                    self._random_delay(1, 2)
                    page.click(link_50_selector)
                    self._random_delay(2, 3)
                    page.wait_for_load_state("domcontentloaded")
                    logger.info("  → Configurado a 50 artículos por página")
                except Exception as e:
                    logger.warning("  → No se pudo cambiar a 50 artículos/página")
                
                # ============================================
                # PASO 6: Extraer artículos
                # ============================================
                max_pages = min((max_results // 50) + 1, 50)
                
                for page_num in range(1, max_pages + 1):
                    logger.info(f"  → Procesando página {page_num}/{max_pages}...")
                    
                    page.wait_for_selector(".search__item", timeout=60000)
                    results = page.query_selector_all(".search__item")
                    
                    for i, result in enumerate(results):
                        try:
                            article = self._extract_article_data(result, page_num, i)
                            if article and article['title'] != "Unknown":
                                articles.append(article)
                                
                                if len(articles) >= max_results:
                                    logger.info(f"  → Alcanzado límite de {max_results} artículos")
                                    break
                        except Exception as e:
                            logger.debug(f"    Error extrayendo artículo {i}: {e}")
                    
                    if len(articles) >= max_results:
                        break
                    
                    # Ir a siguiente página
                    try:
                        next_button = page.query_selector(".pagination__btn--next")
                        if next_button:
                            self._random_delay(2, 4)  # Delay aleatorio entre páginas
                            next_button.click()
                            self._random_delay(3, 5)
                            page.wait_for_load_state("domcontentloaded", timeout=60000)
                        else:
                            logger.info("  → No hay más páginas")
                            break
                    except Exception as e:
                        logger.warning(f"  → Error navegando a página {page_num + 1}: {e}")
                        break
                
                # Guardar cookies al final
                self._save_cookies(context)
                
                browser.close()
                
                elapsed = time.time() - start_time
                logger.info(f"✓ Scraping ACM completado: {len(articles)} artículos en {elapsed:.1f}s")
                
                if articles:
                    df = pd.DataFrame(articles)
                    return df
                else:
                    logger.warning("No se obtuvieron artículos")
                    return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error en scraping de ACM: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _extract_article_data(self, result, page_num: int, index: int) -> dict:
        """Extrae datos de un artículo individual"""
        article = {
            'title': '',
            'authors': '',
            'year': '',
            'abstract': '',
            'keywords': '',
            'doi': '',
            'url': '',
            'venue': '',
            'type': 'article',
            'source': 'ACM'
        }
        
        try:
            # Título
            title_elem = result.query_selector(".hlFld-Title a")
            if title_elem:
                article['title'] = title_elem.inner_text().strip()
                article['url'] = 'https://dl.acm.org' + title_elem.get_attribute("href")
            else:
                return None
            
            # Autores
            authors_elem = result.query_selector(".rlist--inline")
            if authors_elem:
                article['authors'] = authors_elem.inner_text().strip()
            
            # Año
            year_elem = result.query_selector(".bookPubDate")
            if year_elem:
                year_text = year_elem.inner_text()
                year_match = re.search(r'\b(19|20)\d{2}\b', year_text)
                if year_match:
                    article['year'] = year_match.group(0)
            
            # Venue
            venue_elem = result.query_selector(".issue-item__detail")
            if venue_elem:
                article['venue'] = venue_elem.inner_text().split('\n')[0].strip()
            
            # Abstract
            abstract_elem = result.query_selector(".issue-item__abstract")
            if abstract_elem:
                article['abstract'] = abstract_elem.inner_text().strip()
            
            # DOI
            if '/doi/' in article['url']:
                article['doi'] = article['url'].split('/doi/')[-1]
            
            return article
            
        except Exception as e:
            logger.debug(f"Error parseando artículo {page_num}_{index}: {e}")
            return None
    
    def save_to_bibtex(self, df: pd.DataFrame, filepath: str = "data/raw/acm_data.bib"):
        """Guarda DataFrame en formato BibTeX"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                f.write(f"@article{{acm{idx},\n")
                f.write(f"  title = {{{row['title']}}},\n")
                f.write(f"  author = {{{row['authors']}}},\n")
                f.write(f"  year = {{{row['year']}}},\n")
                f.write(f"  journal = {{{row['venue']}}},\n")
                f.write(f"  abstract = {{{row['abstract']}}},\n")
                f.write(f"  doi = {{{row['doi']}}},\n")
                f.write(f"  url = {{{row['url']}}}\n")
                f.write("}\n\n")
        
        logger.info(f"✓ Archivo BibTeX guardado: {filepath}")