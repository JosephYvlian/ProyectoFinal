"""
Scraper para ACM Digital Library
Implementación con Playwright - Acceso vía Universidad del Quindío

"""

import time
import pandas as pd
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ACMScraper:
    """
    Scraper automatizado para ACM Digital Library.
    Acceso vía biblioteca Universidad del Quindío.
    
    Características:
    - Acceso institucional automático
    - Paginación hasta 50 páginas (50 artículos/página)
    - Extracción completa de metadatos
    - Formato BibTeX
    """
    
    def __init__(self, email: str = "josephy.garciac@uqvirtual.edu.co"):
        """
        Inicializa el scraper.
        
        """
        self.email = email
        self.base_url = "https://library.uniquindio.edu.co/databases"
    
    def scrape(self, query: str, max_results: int = 100) -> pd.DataFrame:
        """
        Scrapea artículos de ACM Digital Library.
        
        """
        logger.info(f"Iniciando scraping de ACM: '{query}'")
        logger.info(f"Acceso institucional: {self.email}")
        
    
        from playwright.sync_api import sync_playwright
       
        
        articles = []
        
        try:
            with sync_playwright() as p:
                start_time = time.time()
                
                # Lanzar navegador
                browser = p.chromium.launch(headless=False)  # headless=True para modo silencioso
                page = browser.new_page()
                
                # Paso 1: Acceder a biblioteca UQ
                logger.info("  → Accediendo a biblioteca Universidad del Quindío...")
                page.goto(self.base_url)
                page.wait_for_load_state("domcontentloaded")
                
                # Paso 2: Click en Facultad de Ingeniería
                fac_ingenieria_selector = "div[data-content-listing-item='fac-ingenier-a']"
                page.click(fac_ingenieria_selector)
                page.wait_for_load_state("domcontentloaded")
                
                # Paso 3: Click en ACM Digital Library
                logger.info("  → Accediendo a ACM Digital Library...")
                elements = page.locator("//a[contains(@href, 'dl.acm.org')]//span[contains(text(), 'ACM Digital Library')]")
                count = elements.count()
                
                clicked = False
                for i in range(count):
                    if elements.nth(i).is_visible():
                        elements.nth(i).click()
                        page.wait_for_load_state("domcontentloaded")
                        logger.info("  → Acceso a ACM exitoso")
                        clicked = True
                        break
                
                if not clicked:
                    logger.error("No se pudo acceder a ACM Digital Library")
                    browser.close()
                    return pd.DataFrame()
                
                # Paso 4: Realizar búsqueda
                logger.info(f"  → Buscando: '{query}'...")
                search_selector = "input[name='AllField']"
                page.wait_for_selector(search_selector, timeout=60000)
                page.fill(search_selector, query)
                page.press(search_selector, "Enter")
                page.wait_for_selector(".search__item", timeout=60000)
                
                # Paso 5: Cambiar a 50 artículos por página
                try:
                    link_50_selector = "a[href*='pageSize=50']"
                    page.wait_for_selector(link_50_selector, timeout=10000)
                    page.click(link_50_selector)
                    page.wait_for_load_state("domcontentloaded")
                    logger.info("  → Configurado a 50 artículos por página")
                except Exception as e:
                    logger.warning("  → No se pudo cambiar a 50 artículos/página")
                
                # Paso 6: Extraer artículos de múltiples páginas
                max_pages = (max_results // 50) + 1  # Calcular páginas necesarias
                max_pages = min(max_pages, 50)  # Límite de 50 páginas
                
                for page_num in range(1, max_pages + 1):
                    logger.info(f"  → Procesando página {page_num}/{max_pages}...")
                    
                    # Esperar resultados
                    page.wait_for_selector(".search__item", timeout=60000)
                    results = page.query_selector_all(".search__item")
                    
                    # Extraer artículos de esta página
                    for i, result in enumerate(results):
                        try:
                            article = self._extract_article_data(result, page_num, i)
                            if article and article['title'] != "Unknown":
                                articles.append(article)
                                
                                # Verificar si alcanzamos el máximo
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
                            next_button.click()
                            time.sleep(3)  # Espera entre páginas
                            page.wait_for_load_state("domcontentloaded", timeout=60000)
                        else:
                            logger.info("  → No hay más páginas")
                            break
                    except Exception as e:
                        logger.warning(f"  → Error navegando a página {page_num + 1}: {e}")
                        break
                
                browser.close()
                
                elapsed = time.time() - start_time
                logger.info(f"✓ Scraping ACM completado: {len(articles)} artículos en {elapsed:.1f}s")
                
                # Convertir a DataFrame
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
        """
        Extrae datos de un artículo individual.
        
        """
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
            
            # Venue (revista/conferencia)
            venue_elem = result.query_selector(".issue-item__detail")
            if venue_elem:
                article['venue'] = venue_elem.inner_text().split('\n')[0].strip()
            
            # Abstract
            abstract_elem = result.query_selector(".issue-item__abstract")
            if abstract_elem:
                article['abstract'] = abstract_elem.inner_text().strip()
            
            # DOI (extraer de URL)
            if '/doi/' in article['url']:
                article['doi'] = article['url'].split('/doi/')[-1]
            
            return article
            
        except Exception as e:
            logger.debug(f"Error parseando artículo {page_num}_{index}: {e}")
            return None
    
    def save_to_bibtex(self, df: pd.DataFrame, filepath: str = "data/raw/acm_data.bib"):
        """
        Guarda DataFrame en formato BibTeX.
        
        """
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