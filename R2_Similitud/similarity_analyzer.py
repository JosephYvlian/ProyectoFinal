"""
Requerimiento 2: Analizador de Similitud Textual
Universidad del Quindío - Análisis de Algoritmos

Clase principal que orquesta el análisis de similitud usando
4 algoritmos clásicos + 2 algoritmos IA.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Dict, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SimilarityAnalyzer:
    """
    Analizador de similitud textual con 6 algoritmos.
    
    Algoritmos Clásicos:
    1. Levenshtein (distancia de edición)
    2. Jaccard (similitud de conjuntos)
    3. Cosine + TF-IDF (vectorización estadística)
    4. Dice (coeficiente de Dice)
    
    Algoritmos IA:
    5. BERT (embeddings con transformers)
    6. Sentence-BERT (optimizado para similitud)
    """
    
    def __init__(self):
        """Inicializa el analizador."""
        self.algorithms = None
        self._load_algorithms()
    
    def _load_algorithms(self):
        """Carga todos los algoritmos disponibles."""
        from .algorithms.classic import (
            levenshtein_similarity,
            jaccard_similarity,
            cosine_tfidf_similarity,
            dice_coefficient
        )
        from .algorithms.ai_based import (
            bert_similarity,
            sentence_bert_similarity
        )
        
        self.algorithms = {
            'Levenshtein': {
                'function': levenshtein_similarity,
                'type': 'Clásico',
                'category': 'Distancia de Edición'
            },
            'Jaccard': {
                'function': jaccard_similarity,
                'type': 'Clásico',
                'category': 'Similitud de Conjuntos'
            },
            'Cosine_TFIDF': {
                'function': cosine_tfidf_similarity,
                'type': 'Clásico',
                'category': 'Vectorización Estadística'
            },
            'Dice': {
                'function': dice_coefficient,
                'type': 'Clásico',
                'category': 'Similitud de Conjuntos'
            },
            'BERT': {
                'function': bert_similarity,
                'type': 'IA',
                'category': 'Deep Learning'
            },
            'SentenceBERT': {
                'function': sentence_bert_similarity,
                'type': 'IA',
                'category': 'Deep Learning Optimizado'
            }
        }
    
    def analyze_texts(self, texts: List[str], 
                     titles: List[str] = None,
                     algorithms: List[str] = None) -> Dict:
        """
        Analiza similitud entre múltiples textos.
        
        Args:
            texts: Lista de textos (abstracts) a comparar
            titles: Lista de títulos correspondientes (opcional)
            algorithms: Lista de algoritmos a usar (None = todos)
        
        Returns:
            Dict con resultados completos
        """
        n = len(texts)
        
        if n < 2:
            raise ValueError("Se requieren al menos 2 textos para comparar")
        
        logger.info("="*70)
        logger.info("REQUERIMIENTO 2: ANÁLISIS DE SIMILITUD TEXTUAL")
        logger.info("="*70)
        logger.info(f"\nAnalizando {n} textos con {len(self.algorithms)} algoritmos...")
        
        # Si no se especifican algoritmos, usar todos
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        
        # Generar todos los pares posibles
        pairs = list(combinations(range(n), 2))
        logger.info(f"Total de comparaciones: {len(pairs)}")
        
        # Estructura de resultados
        results = {
            'texts': texts,
            'titles': titles or [f"Texto {i+1}" for i in range(n)],
            'n_texts': n,
            'n_comparisons': len(pairs),
            'pairs': pairs,
            'similarities': {},
            'details': {}
        }
        
        # Ejecutar cada algoritmo
        for algo_name in algorithms:
            if algo_name not in self.algorithms:
                logger.warning(f"Algoritmo '{algo_name}' no encontrado, saltando...")
                continue
            
            algo_info = self.algorithms[algo_name]
            logger.info(f"\n→ Ejecutando: {algo_name} ({algo_info['type']})")
            
            similarities = []
            details_list = []
            
            for i, j in pairs:
                try:
                    # Ejecutar algoritmo
                    similarity, details = algo_info['function'](texts[i], texts[j])
                    
                    similarities.append(similarity)
                    details_list.append(details)
                    
                    logger.debug(f"  Par ({i},{j}): {similarity:.3f}")
                    
                except Exception as e:
                    logger.error(f"  Error en par ({i},{j}): {e}")
                    similarities.append(np.nan)
                    details_list.append({'error': str(e)})
            
            results['similarities'][algo_name] = similarities
            results['details'][algo_name] = details_list
            
            # Estadísticas
            valid_sims = [s for s in similarities if not np.isnan(s)]
            if valid_sims:
                logger.info(f"  ✓ Media: {np.mean(valid_sims):.3f}")
                logger.info(f"  ✓ Min: {np.min(valid_sims):.3f}, Max: {np.max(valid_sims):.3f}")
        
        # Crear resumen estadístico
        results['statistics'] = self._calculate_statistics(results)
        
        logger.info("\n" + "="*70)
        logger.info("ANÁLISIS COMPLETADO")
        logger.info("="*70)
        
        return results
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """Calcula estadísticas por algoritmo."""
        stats = {}
        
        for algo_name, similarities in results['similarities'].items():
            valid_sims = [s for s in similarities if not np.isnan(s)]
            
            if valid_sims:
                stats[algo_name] = {
                    'mean': np.mean(valid_sims),
                    'std': np.std(valid_sims),
                    'min': np.min(valid_sims),
                    'max': np.max(valid_sims),
                    'median': np.median(valid_sims),
                    'count': len(valid_sims)
                }
            else:
                stats[algo_name] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'median': np.nan,
                    'count': 0
                }
        
        return stats
    
    
    def save_results(self, results: Dict, output_dir: str = "data/outputs"):
        """
        Guarda resultados en archivos.
        
        Args:
            results: Resultados del análisis
            output_dir: Directorio de salida
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        
        # Guardar detalles completos (JSON)
        import json
        
        # Preparar datos para JSON (convertir numpy a python)
        json_data = {
            'n_texts': results['n_texts'],
            'n_comparisons': results['n_comparisons'],
            'titles': results['titles'],
            'similarities': {
                algo: [float(s) if not np.isnan(s) else None 
                       for s in sims]
                for algo, sims in results['similarities'].items()
            },
            'statistics': {
                algo: {k: float(v) if not np.isnan(v) else None 
                       for k, v in stat.items()}
                for algo, stat in results['statistics'].items()
            }
        }
        
        json_path = output_path / 'similarity_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Resultados completos guardados: {json_path}")

    
    def print_summary(self, results: Dict):
        """Imprime resumen legible de resultados."""
        print("\n" + "="*70)
        print("RESUMEN DE SIMILITUDES")
        print("="*70)
        
        # Estadísticas
        print("\n" + "="*70)
        print("ESTADÍSTICAS POR ALGORITMO")
        print("="*70)
        
        stats_df = pd.DataFrame(results['statistics']).T
        print(stats_df.round(3).to_string())
    
        
    def visualize_results(self, results: Dict, save_path: str = None):
        """
        Crea visualización de resultados (barras por promedio de similitud).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from matplotlib.patches import Patch
        
        # Crear figura y ejes (solo uno por ahora)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # DataFrame de estadísticas
        stats_df = pd.DataFrame(results["statistics"]).T
        stats_df = stats_df.sort_values("mean", ascending=True)

        # Colores según tipo de algoritmo
        colors = [
            "green" if algo.lower() in ["bert", "sentencebert", "bert-base", "sbert"] else "blue"
            for algo in stats_df.index
        ]

        # Gráfico barras horizontales
        ax.barh(stats_df.index, stats_df["mean"], color=colors, alpha=0.8)
        ax.set_xlabel("Similitud Promedio")
        ax.set_title("Comparación de Algoritmos por Media de Similitud",
                    fontsize=14, fontweight="bold")
        ax.grid(axis='x', linestyle="--", alpha=0.4)

        # Leyenda
        legend_elements = [
            Patch(facecolor='blue', alpha=0.8, label='Métodos Clásicos'),
            Patch(facecolor='green', alpha=0.8, label='Métodos IA')
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        # Guardar si corresponde
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
        return ax

    
    def select_articles_interactive(self, df: pd.DataFrame, 
                                   n_articles: int = 3) -> pd.DataFrame:
        """
        Selección interactiva de artículos.
        
        """
        # Filtrar artículos con abstract
        df_filtered = df[df['abstract'].notna() & (df['abstract'].str.len() > 50)]
        
        if len(df_filtered) < n_articles:
            print(f"⚠ Solo hay {len(df_filtered)} artículos con abstract")
            return df_filtered
        
        print("\n" + "="*70)
        print(f"SELECCIÓN DE {n_articles} ARTÍCULOS")
        print("="*70)
        print("\nMétodos disponibles:")
        print("[1] Aleatorio")
        print("[2] Manual (lista)")
        
        modo = input("\nSeleccione método (1/2): ").strip()
        
        if modo == "2":
            # Manual
            display_df = df_filtered.head(20)
            
            print("\n" + "-"*70)
            for idx, row in display_df.iterrows():
                print(f"[{idx}] {row['title'][:65]}...")
            print("-"*70)
            
            seleccion = input(f"\nSeleccione {n_articles} (separados por coma): ").strip()
            indices = [int(x.strip()) for x in seleccion.split(',')]
            
            return df_filtered.iloc[indices]
        
        elif modo == "1":
            # Aleatorio
            return df_filtered.sample(n_articles, random_state=42)
        
    
    def get_algorithm_info(self, algorithm_name: str) -> str:
        """
        Retorna información detallada de un algoritmo.
        
        Args:
            algorithm_name: Nombre del algoritmo
        
        Returns:
            Información como string
        """
        if algorithm_name not in self.algorithms:
            return f"Algoritmo '{algorithm_name}' no encontrado"
        
        algo = self.algorithms[algorithm_name]
        func = algo['function']
        
        # Obtener docstring
        info = f"\n{'='*70}\n"
        info += f"ALGORITMO: {algorithm_name}\n"
        info += f"Tipo: {algo['type']}\n"
        info += f"Categoría: {algo['category']}\n"
        info += f"{'='*70}\n\n"
        info += func.__doc__ if func.__doc__ else "Sin documentación"
        
        return info