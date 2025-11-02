"""
Scraper para IEEE Xplore
Implementación con Playwright - Acceso vía Universidad del Quindío
Requiere autenticación con Google (correo institucional)
"""

import time
import pandas as pd
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class IEEEScraper:
    """
    Scraper automatizado para IEEE Xplore.
    Acceso vía biblioteca Universidad del Quindío con login Google.
    
    Características:
    - Login automático con Google
    - Hasta 45 páginas (100 artículos/página)
    - Extracción completa de metadatos
    """
    
    def __init__(self, email: str = "josephy.garciac@uqvirtual.edu.co", 
                 password: str = None):
        """
        Inicializa el scraper.
        
        """
        self.email = email
        self.password = password
        self.base_url = "https://library.uniquindio.edu.co/databases"
    
    def scrape(self, query: str, max_results: int = 100) -> pd.DataFrame:
        """
        Scrapea artículos de IEEE Xplore.
        
        """
        logger.info(f"Iniciando scraping de IEEE: '{query}'")
        
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.error("Playwright no está instalado")
            return pd.DataFrame()
        
        articles = []
        
        try:
            with sync_playwright() as p:
                start_time = time.time()
                
                browser = p.chromium.launch(headless=False)
                page = browser.new_page()
                
                # Paso 1: Acceder a biblioteca UQ
                logger.info("  → Accediendo a biblioteca Universidad del Quindío...")
                page.goto(self.base_url)
                page.wait_for_load_state("domcontentloaded")
                
                # Paso 2: Click en Facultad de Ingeniería
                fac_ingenieria_selector = "div[data-content-listing-item='fac-ingenier-a']"
                page.click(fac_ingenieria_selector)
                page.wait_for_load_state("domcontentloaded")
                
                # Paso 3: Click en IEEE Explorer
                logger.info("  → Accediendo a IEEE Xplore...")
                elements = page.locator("//a[contains(@href, 'ieeexplore-ieee-org')]//span[contains(text(), 'IEEE')]")
                count = elements.count()
                
                clicked = False
                for i in range(count):
                    if elements.nth(i).is_visible():
                        elements.nth(i).click()
                        page.wait_for_load_state("domcontentloaded")
                        clicked = True
                        break
                
                if not clicked:
                    logger.error("No se pudo acceder a IEEE")
                    browser.close()
                    return pd.DataFrame()
                
                # Paso 4: Login con Google
                logger.info("  → Iniciando sesión con Google...")
                try:
                    google_login_button = "a#btn-google"
                    page.click(google_login_button)
                    
                    # Ingresar email
                    email_input = "input#identifierId"
                    page.fill(email_input, self.email)
                    page.click("button:has-text('Siguiente')")
                    page.wait_for_load_state("domcontentloaded")
                    
                    # Ingresar contraseña
                    if self.password:
                        password_input = "input[name='Passwd']"
                        page.fill(password_input, self.password)
                        page.click("button:has-text('Siguiente')")
                        page.wait_for_load_state("domcontentloaded")
                        logger.info("  → Login exitoso")
                    else:
                        logger.warning("  → Contraseña no proporcionada. Ingrese manualmente.")
                        input("Presione Enter después de iniciar sesión manualmente...")
                
                except Exception as e:
                    logger.warning(f"  → Error en login automático: {e}")
                    logger.info("  → Ingrese credenciales manualmente...")
                    input("Presione Enter después de iniciar sesión...")
                
                # Paso 5: Realizar búsqueda
                logger.info(f"  → Buscando: '{query}'...")
                search_selector = 'input[type="search"]'
                page.wait_for_selector(search_selector, timeout=60000)
                page.fill(search_selector, query)
                page.press(search_selector, "Enter")
                page.wait_for_selector(".List-results-items", timeout=60000)
                
                # Paso 6: Cambiar a 100 resultados por página
                try:
                    items_per_page_button = page.locator('button:has-text("Items Per Page")')
                    items_per_page_button.click()
                    option_100 = page.locator('button:has-text("100")')
                    option_100.click()
                    page.wait_for_timeout(5000)
                    logger.info("  → Configurado a 100 artículos por página")
                except Exception as e:
                    logger.warning("  → No se pudo cambiar a 100 artículos/página")
                
                # Paso 7: Extraer artículos
                max_pages = min((max_results // 100) + 1, 45)
                current_page = 1
                
                while current_page <= max_pages and len(articles) < max_results:
                    logger.info(f"  → Procesando página {current_page}/{max_pages}...")
                    
                    page.wait_for_selector(".List-results-items", timeout=60000)
                    results = page.query_selector_all(".List-results-items")
                    
                    for i, result in enumerate(results):
                        try:
                            article = self._extract_article_data(result, current_page, i)
                            if article and article['title'] != "Unknown":
                                articles.append(article)
                                
                                if len(articles) >= max_results:
                                    break
                        except Exception as e:
                            logger.debug(f"    Error extrayendo artículo: {e}")
                    
                    if len(articles) >= max_results:
                        break
                    
                    # Navegar a siguiente página
                    try:
                        if current_page in [10, 20, 30, 40]:
                            # Cargar siguiente bloque de 10 páginas
                            next_button = page.locator('li.next-page-set button:has-text("Next")')
                            if next_button.is_visible():
                                next_button.click()
                                page.wait_for_timeout(5000)
                                page.wait_for_selector(".List-results-items", timeout=60000)
                            else:
                                break
                        else:
                            # Ir a página específica
                            next_page_button = page.locator(f'li button.stats-Pagination_{current_page + 1}')
                            if next_page_button.is_visible():
                                next_page_button.click()
                                page.wait_for_timeout(5000)
                                page.wait_for_selector(".List-results-items", timeout=60000)
                            else:
                                break
                        
                        current_page += 1
                        
                    except Exception as e:
                        logger.warning(f"  → Error navegando a página {current_page + 1}: {e}")
                        break
                
                browser.close()
                
                elapsed = time.time() - start_time
                logger.info(f"✓ Scraping IEEE completado: {len(articles)} artículos en {elapsed:.1f}s")
                
                if articles:
                    df = pd.DataFrame(articles)
                    return df
                else:
                    return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error en scraping de IEEE: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _extract_article_data(self, result, page_num: int, index: int) -> dict:
        """Extrae datos de un artículo."""
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
            'source': 'IEEE'
        }
        
        try:
            # Título y URL
            title_elem = result.query_selector("a.fw-bold")
            if title_elem:
                article['title'] = title_elem.inner_text().strip()
                link = title_elem.get_attribute("href")
                article['url'] = f"https://ieeexplore.ieee.org{link}"
            else:
                return None
            
            # Autores
            author_elem = result.query_selector(".text-base-md-lh")
            if author_elem:
                article['authors'] = author_elem.inner_text().replace("\n", " ").strip()
            
            # Año
            year_elem = result.query_selector(".publisher-info-container")
            if year_elem:
                year_text = year_elem.inner_text()
                year_match = re.search(r'\b(19|20)\d{2}\b', year_text)
                if year_match:
                    article['year'] = year_match.group(0)
            
            # Venue (revista/conferencia)
            journal_elem = result.query_selector("div.description > a[xplhighlight]")
            if journal_elem:
                article['venue'] = journal_elem.inner_text().strip()
            
            # Tipo de publicación
            tipo_elements = result.query_selector_all("span[xplhighlight]")
            for element in tipo_elements:
                text = element.inner_text().strip()
                if not ("Year:" in text or "Volume:" in text or "Issue:" in text or re.search(r'\d', text)):
                    article['type'] = text.lower()
                    break
            
            # Abstract
            abstract_elem = result.query_selector(".twist-container")
            if abstract_elem:
                article['abstract'] = abstract_elem.inner_text().strip()
            
            return article
            
        except Exception as e:
            logger.debug(f"Error parseando artículo {page_num}_{index}: {e}")
            return None
    
    def save_to_bibtex(self, df: pd.DataFrame, filepath: str = "data/raw/ieee_data.bib"):
        """Guarda en formato BibTeX."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                f.write(f"@article{{ieee{idx},\n")
                f.write(f"  title = {{{row['title']}}},\n")
                f.write(f"  author = {{{row['authors']}}},\n")
                f.write(f"  year = {{{row['year']}}},\n")
                f.write(f"  journal = {{{row['venue']}}},\n")
                f.write(f"  tipo = {{{row['type']}}},\n")
                f.write(f"  abstract = {{{row['abstract']}}},\n")
                f.write(f"  url = {{{row['url']}}}\n")
                f.write("}\n\n")
        
        logger.info(f"✓ Archivo BibTeX guardado: {filepath}")