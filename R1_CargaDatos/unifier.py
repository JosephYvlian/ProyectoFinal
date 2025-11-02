"""
Requerimiento 1: Automatización de Descarga y Unificación de Datos

Clase principal que orquesta:
1. Descarga automatizada de ACM y IEEE
2. Unificación de datos
3. Detección y eliminación de duplicados
4. Generación de archivos de salida
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class DataUnifier:
    """
    Clase principal para el Requerimiento 1.

    """
    
    def __init__(self, similarity_threshold: float = 0.85, 
                 email: str = None, 
                 password: str = None):
        """
        Inicializa el unificador.
        
        """
        self.similarity_threshold = similarity_threshold
        self.output_dir = Path('data/processed')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración para scrapers
        self.config = {
            'email': email or 'josephy.garciac@uqvirtual.edu.co',
            'password': password
        }
        
    def execute_pipeline(self, query: str = "generative artificial intelligence", 
                        max_results: int = 100,
                        auto_download: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
        logger.info("="*70)
        logger.info("INICIANDO REQUERIMIENTO 1: AUTOMATIZACIÓN Y UNIFICACIÓN")
        logger.info("="*70)
        
        # Paso 1: Obtener datos
        if auto_download:
            dataframes = self._download_data(query, max_results)
        else:
            dataframes = self._load_existing_data()
        
        if not dataframes:
            raise ValueError("No se pudieron obtener datos. Verifique archivos o conexión.")
        
        # Paso 2: Unificar
        logger.info("\n[2/4] Unificando datos de múltiples fuentes...")
        combined_df = self._combine_dataframes(dataframes)
        logger.info(f"Total de registros combinados: {len(combined_df)}")
        
        # Paso 3: Detectar y eliminar duplicados
        logger.info("\n[3/4] Detectando duplicados...")
        unified_df, duplicates_df = self._remove_duplicates(combined_df)
        logger.info(f"Registros únicos: {len(unified_df)}")
        logger.info(f"Duplicados encontrados: {len(duplicates_df)}")
        
        # Paso 4: Guardar resultados
        logger.info("\n[4/4] Guardando resultados...")
        self._save_results(unified_df, duplicates_df)
        
        # Resumen final
        self._print_summary(combined_df, unified_df, duplicates_df)
        
        return unified_df, duplicates_df
    
    def _download_data(self, query: str, max_results: int) -> List[pd.DataFrame]:
        """
        Descarga datos automáticamente de ACM y IEEE.
        
        """
        logger.info("\n[1/4] Descargando datos automáticamente...")
        
        dataframes = []
        
        # Importar scrapers
        try:
            from .scrapers.acm_scraper import ACMScraper
            from .scrapers.ieee_scraper import IEEEScraper
        except ImportError:
            logger.error("Error importando scrapers. Verifique instalación.")
            logger.error("Instalar Playwright: pip install playwright && playwright install")
            return []
        
        # Scraper ACM
        logger.info("\n  → Descargando de ACM Digital Library...")
        try:
            acm_scraper = ACMScraper(email=self.config.get('email'))
            df_acm = acm_scraper.scrape(query, max_results)
            
            if df_acm is not None and len(df_acm) > 0:
                logger.info(f"ACM: {len(df_acm)} artículos descargados")
                
                # Guardar raw data
                raw_dir = Path('data/raw')
                raw_dir.mkdir(parents=True, exist_ok=True)
                df_acm.to_csv(raw_dir / 'acm_data.csv', index=False)
                
                dataframes.append(df_acm)
            else:
                logger.warning("ACM: No se obtuvieron datos")
        except Exception as e:
            logger.error(f"Error en ACM: {str(e)}")
        
        # Scraper IEEE
        logger.info("\n  → Descargando de IEEE Xplore...")
        try:
            email = self.config.get('email')
            password = self.config.get('password')
            
            ieee_scraper = IEEEScraper(email=email, password=password)
            df_ieee = ieee_scraper.scrape(query, max_results)
            
            if df_ieee is not None and len(df_ieee) > 0:
                logger.info(f"IEEE: {len(df_ieee)} artículos descargados")
                
                # Guardar raw data
                raw_dir = Path('data/raw')
                raw_dir.mkdir(parents=True, exist_ok=True)
                df_ieee.to_csv(raw_dir / 'ieee_data.csv', index=False)
                
                dataframes.append(df_ieee)
            else:
                logger.warning("IEEE: No se obtuvieron datos")
        except Exception as e:
            logger.error(f"Error en IEEE: {str(e)}")
        
        return dataframes
    
    def _load_existing_data(self) -> List[pd.DataFrame]:
        """
        Carga datos desde archivos CSV existentes.
        
        """
        logger.info("\n[1/4] Cargando datos desde archivos existentes...")
        
        dataframes = []
        raw_dir = Path('data/raw')
        
        # ACM
        acm_path = raw_dir / 'acm_data.csv'
        if acm_path.exists():
            try:
                df_acm = pd.read_csv(acm_path)
                logger.info(f"ACM: {len(df_acm)} registros cargados")
                dataframes.append(df_acm)
            except Exception as e:
                logger.error(f"Error cargando ACM: {e}")
        else:
            logger.warning(f"No se encontró: {acm_path}")
        
        # IEEE
        ieee_path = raw_dir / 'ieee_data.csv'
        if ieee_path.exists():
            try:
                df_ieee = pd.read_csv(ieee_path)
                logger.info(f"IEEE: {len(df_ieee)} registros cargados")
                dataframes.append(df_ieee)
            except Exception as e:
                logger.error(f"Error cargando IEEE: {e}")
        else:
            logger.warning(f"No se encontró: {ieee_path}")
        
        return dataframes
    
    def _combine_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combina múltiples DataFrames y estandariza columnas.
        
        """
        if not dataframes:
            raise ValueError("No hay DataFrames para combinar")
        
        # Concatenar todos
        combined = pd.concat(dataframes, ignore_index=True)
        
        # Estandarizar columnas
        combined = self._standardize_columns(combined)
        
        # Limpiar datos
        combined = self._clean_data(combined)
        
        return combined
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estandariza nombres y asegura existencia de columnas requeridas.
        
        Columnas estándar:
        - title: Título del trabajo
        - authors: Autores (separados por ;)
        - year: Año de publicación
        - abstract: Resumen
        - keywords: Palabras clave (separadas por ;)
        - doi: Digital Object Identifier
        - url: URL del artículo
        - venue: Revista/Conferencia
        - type: article, conference, etc.
        - source: ACM, IEEE, etc.
        """
        standard_columns = [
            'title', 'authors', 'year', 'abstract', 'keywords',
            'doi', 'url', 'venue', 'type', 'source'
        ]
        
        # Crear columnas faltantes
        for col in standard_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Seleccionar solo columnas estándar
        df = df[standard_columns].copy()
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y normaliza los datos.

        """
        logger.info("  → Limpiando datos...")
        
        # Reemplazar NaN con string vacío
        df = df.fillna('')
        
        # Limpiar strings
        string_columns = ['title', 'authors', 'abstract', 'keywords', 'venue']
        for col in string_columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        
        # Normalizar año
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        
        # Eliminar registros sin título
        initial_count = len(df)
        df = df[df['title'].str.len() > 0].copy()
        removed = initial_count - len(df)
        
        if removed > 0:
            logger.info(f"  → Eliminados {removed} registros sin título")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detecta y elimina duplicados basándose en similitud de títulos.

        """
        # Crear título normalizado
        df['title_norm'] = df['title'].str.lower().str.strip()
        df['title_norm'] = df['title_norm'].str.replace(r'[^\w\s]', '', regex=True)
        
        # 1. Detectar duplicados exactos
        logger.info("  → Buscando duplicados exactos...")
        exact_dup_mask = df.duplicated(subset=['title_norm'], keep='first')
        n_exact = exact_dup_mask.sum()
        logger.info(f"Duplicados exactos: {n_exact}")
        
        exact_duplicates = df[exact_dup_mask].copy()
        potential_uniques = df[~exact_dup_mask].copy()
        
        # 2. Detectar duplicados por similitud
        logger.info("  → Buscando duplicados por similitud...")
        logger.info(f"    Umbral: {self.similarity_threshold}")
        
        fuzzy_duplicates = []
        unique_indices = []
        seen_titles = []
        
        for idx, row in potential_uniques.iterrows():
            title = row['title_norm']
            
            is_duplicate = False
            for seen_title, seen_idx in seen_titles:
                similarity = self._calculate_similarity(title, seen_title)
                
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    fuzzy_duplicates.append(idx)
                    logger.debug(f"    Duplicado fuzzy ({similarity:.2f}): '{title[:50]}...'")
                    break
            
            if not is_duplicate:
                unique_indices.append(idx)
                seen_titles.append((title, idx))
        
        n_fuzzy = len(fuzzy_duplicates)
        logger.info(f"Duplicados por similitud: {n_fuzzy}")
        
        # Crear DataFrames finales
        unique_df = potential_uniques.loc[unique_indices].copy()
        fuzzy_dup_df = potential_uniques.loc[fuzzy_duplicates].copy()
        
        all_duplicates = pd.concat([exact_duplicates, fuzzy_dup_df], ignore_index=True)
        
        # Limpiar columnas temporales
        unique_df = unique_df.drop(columns=['title_norm'])
        all_duplicates = all_duplicates.drop(columns=['title_norm'])
        
        unique_df = self._consolidate_information(unique_df)
        
        return unique_df.reset_index(drop=True), all_duplicates.reset_index(drop=True)
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calcula similitud entre dos cadenas (Ratcliff-Obershelp)."""
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _consolidate_information(self, df: pd.DataFrame) -> pd.DataFrame:
        df['completeness'] = df.apply(
            lambda row: sum([1 for val in row if val != '']), axis=1
        )
        df = df.sort_values('completeness', ascending=False)
        df = df.drop(columns=['completeness'])
        return df
    
    def _save_results(self, unified_df: pd.DataFrame, duplicates_df: pd.DataFrame):
        """Guarda resultados en archivos CSV."""
        unified_path = self.output_dir / 'unified_data.csv'
        duplicates_path = self.output_dir / 'duplicates.csv'
        
        unified_df.to_csv(unified_path, index=False)
        duplicates_df.to_csv(duplicates_path, index=False)
        
        logger.info(f"Datos unificados → {unified_path}")
        logger.info(f"Duplicados → {duplicates_path}")
        
        logger.info("\n" + "="*70)
        logger.info("✓✓✓ REQUERIMIENTO 1 COMPLETADO ✓✓✓")
        logger.info("="*70 + "\n")

    def _print_summary(self, combined_df: pd.DataFrame, unified_df: pd.DataFrame, duplicates_df: pd.DataFrame):
        """Imprime resumen de resultados."""
        total = len(combined_df)
        unique = len(unified_df)
        duplicates = len(duplicates_df)
        dedup_rate = (duplicates / total * 100) if total > 0 else 0
        
        logger.info("\n" + "="*70)
        logger.info("RESUMEN REQUERIMIENTO 1")
        logger.info("="*70)
        logger.info(f"Total de registros combinados:  {total:>6}")
        logger.info(f"Registros únicos:               {unique:>6}")
        logger.info(f"Duplicados eliminados:          {duplicates:>6}")
        logger.info("="*70)