#!/usr/bin/env python3
"""
Sistema de Análisis Bibliométrico con Menú Interactivo
Universidad del Quindío - Análisis de Algoritmos

Permite ejecutar requerimientos de forma independiente sin detener el programa.
El usuario puede ir ejecutando los análisis en el orden que desee.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Estado global de la aplicación
class AppState:
    """Mantiene el estado de los datos cargados entre ejecuciones"""
    def __init__(self):
        self.df = None  # DataFrame con datos unificados
        self.duplicates_df = None
        self.similarity_results = None
        self.frequency_results = None
        self.clustering_results = None
        
    def has_data(self):
        """Verifica si hay datos cargados"""
        return self.df is not None and len(self.df) > 0


# Instancia global
state = AppState()


def clear_screen():
    """Limpia la pantalla de la consola"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Imprime el encabezado del menú"""
    print("\n" + "="*70)
    print("    SISTEMA DE ANÁLISIS BIBLIOMÉTRICO")
    print("    Universidad del Quindío - Análisis de Algoritmos")
    print("    Dominio: Inteligencia Artificial Generativa")
    print("    Joseph Yulian Garcia - Juan Camilo Sánchez")
    print("="*70)


def print_menu():
    """Imprime el menú principal"""
    print("\n" + "-"*70)
    print("  MENÚ PRINCIPAL")
    print("-"*70)
    print("\n[1] Req 1: Cargar y Unificar Datos")
    print("[2] Req 2: Análisis de Similitud Textual")
    print("[3] Req 3: Análisis de Frecuencias de Palabras")
    print("[4] Req 4: Clustering Jerárquico")
    print("[5] Req 5: Visualizaciones y Exportación PDF")
    print("\n[6] Ver resumen de datos cargados")
    print("[7] Ver resultados guardados")
    print("\n[0] Salir")
    print("-"*70)
    
    # Mostrar estado
    if state.has_data():
        print(f"\n✓ Datos cargados: {len(state.df)} artículos")
    else:
        print("\nNo hay datos cargados (Se debe ejecutar el REQUERIMIENTO 1)")
    print()


def pause():
    """Pausa y espera a que el usuario presione Enter"""
    input("\nPresione Enter para continuar...")


# ========== REQUERIMIENTO 1 ==========
def ejecutar_req1():
    """
    Requerimiento 1: Carga y Unificación de Datos
    Automatizado con descarga de ACM y IEEE
    """
    print("\n" + "="*70)
    print("REQUERIMIENTO 1: AUTOMATIZACIÓN Y UNIFICACIÓN DE DATOS")
    print("="*70)
    
    print("\nOpciones:")
    print("[1] Descarga AUTOMATIZADA (scrapea ACM y IEEE)")
    print("[2] Usar archivos EXISTENTES (data/raw/csv)")
    print()
    
    opcion = input("Seleccione opción (1/2): ").strip() or "1"
    
    auto_download = (opcion == "1")
    
    if auto_download:
        print("\n INICIANDO SCRAPING")

    try:
        # Importar y ejecutar Req 1
        from R1_CargaDatos import DataUnifier
        
        # Parámetros
        query = "generative artificial intelligence"
        max_results = 100
        threshold = 0.85
        
        print()
        # Ejecutar
        unifier = DataUnifier(similarity_threshold=threshold)

        unified_df, duplicates_df = unifier.execute_pipeline(
            query=query,
            max_results=max_results,
            auto_download=auto_download
        )
        
        # Actualizar estado
        state.df = unified_df
        state.duplicates_df = duplicates_df
        print("\n" + "="*70)
        print("✓✓✓ REQUERIMIENTO 1 COMPLETADO ✓✓✓")
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        logger.error(f"Error en Req 1: {e}")
        import traceback
        traceback.print_exc()


# ========== REQUERIMIENTO 2 ==========
def ejecutar_req2():
    """
    Requerimiento 2: Análisis de Similitud Textual
    6 algoritmos: 4 clásicos + 2 IA
    """
    print("\n" + "="*70)
    print("REQUERIMIENTO 2: ANÁLISIS DE SIMILITUD TEXTUAL")
    print("="*70)
    
    # Verificar que hay datos
    if not state.has_data():
        print("\nERROR: No hay datos cargados.")
        print("Por favor ejecute el Requerimiento 1 primero.")
        return
    
    print(f"\nDatos disponibles: {len(state.df)} artículos")
    
    # Filtrar artículos con abstract
    df_with_abstract = state.df[
        state.df['abstract'].notna() & 
        (state.df['abstract'].str.len() > 50)
    ]
    
    if len(df_with_abstract) == 0:
        print("\nERROR: No hay artículos con abstract válido.")
        print("Los datos cargados no tienen abstracts suficientemente largos.")
        return
    
    print(f"Artículos con abstract válido: {len(df_with_abstract)}")
    
    # Información sobre algoritmos
    print("\n" + "-"*70)
    print("ALGORITMOS DISPONIBLES:")
    print("-"*70)
    print("\nClásicos:")
    print("  [1] Levenshtein - Distancia de edición")
    print("  [2] Jaccard - Similitud de conjuntos")
    print("  [3] Cosine + TF-IDF - Vectorización estadística")
    print("  [4] Dice - Coeficiente de Sørensen-Dice")
    print("\nInteligencia Artificial:")
    print("  [5] BERT - Embeddings con Transformers")
    print("  [6] Sentence-BERT - Optimizado para similitud")
    print()
    
    # Preguntar configuración
    n_articles_str = input("¿Cuántos artículos comparar? (n): ").strip()
    n_articles = int(n_articles_str)
    n_articles = max(2, min(20, n_articles))
    
    if n_articles > len(df_with_abstract):
        n_articles = len(df_with_abstract)
        print(f"Ajustado a {n_articles} (máximo disponible)")
    
    # Preguntar modo de selección
    print("\n" + "-"*70)
    print("MODO DE SELECCIÓN:")
    print("-"*70)
    print("[1] Aleatorio (rápido)")
    print("[2] Manual (seleccionar de una lista)")
    print()
    
    modo = input("Seleccione modo (1/2): ").strip()
    
    # Preguntar si usar todos los algoritmos
    usar_todos = input("\n¿Usar TODOS los algoritmos? (s/n): ").strip().lower()
    
    algorithms = None
    if usar_todos != 's':
        print("\nSeleccione algoritmos (separados por coma, ej: 1,2,5):")
        seleccion = input("Algoritmos: ").strip()
        
        if seleccion:
            indices = [int(x.strip()) for x in seleccion.split(',') if x.strip().isdigit()]
            algo_map = {
                1: 'Levenshtein',
                2: 'Jaccard',
                3: 'Cosine_TFIDF',
                4: 'Dice',
                5: 'BERT',
                6: 'SentenceBERT'
            }
            algorithms = [algo_map[i] for i in indices if i in algo_map]
    
    try:
        # Importar y ejecutar
        from R2_Similitud import SimilarityAnalyzer
        
        print(f"\n→ Seleccionando {n_articles} artículos aleatoriamente...")
        
        # Seleccionar artículos según modo
        if modo == "2":
            # MODO MANUAL
            print("\n" + "="*70)
            print("SELECCIÓN MANUAL DE ARTÍCULOS")
            print("="*70)
            print(f"\nMostrando primeros 20 artículos con abstract:")
            print("-"*70)
            
            # Mostrar lista de artículos
            display_df = df_with_abstract.head(20).reset_index(drop=True)
            
            for idx, row in display_df.iterrows():
                title_short = row['title'][:60] + "..." if len(row['title']) > 60 else row['title']
                year = row['year'] if 'year' in row and row['year'] else 'N/A'
                source = row['source'] if 'source' in row else 'N/A'
                print(f"[{idx}] {title_short}")
                print(f"    ({source}, {year})")
                print()
            
            print("-"*70)
            print(f"\nSeleccione {n_articles} artículos (números separados por coma)")
            print("Ejemplo: 0,3,7")
            print()
            
            while True:
                seleccion = input("Artículos: ").strip()
                
                try:
                    indices = [int(x.strip()) for x in seleccion.split(',')]
                    
                    # Validar
                    if len(indices) != n_articles:
                        print(f"⚠ Debe seleccionar exactamente {n_articles} artículos")
                        continue
                    
                    if any(i < 0 or i >= len(display_df) for i in indices):
                        print(f"⚠ Índices válidos: 0-{len(display_df)-1}")
                        continue
                    
                    if len(set(indices)) != len(indices):
                        print("⚠ No puede seleccionar el mismo artículo dos veces")
                        continue
                    
                    # Selección válida
                    selected = display_df.iloc[indices]
                    break
                    
                except ValueError:
                    print("⚠ Formato inválido. Use números separados por coma (ej: 0,3,7)")
            
            print("\n✓ Artículos seleccionados:")
            for i, (_, row) in enumerate(selected.iterrows(), 1):
                print(f"  {i}. {row['title'][:70]}...")
        
        else:
            # MODO ALEATORIO
            selected = df_with_abstract.sample(n_articles, random_state=42)
        titles = selected['title'].tolist()
        abstracts = selected['abstract'].tolist()
        
        if modo != "2":
            print("\nArtículos seleccionados (aleatorio):")
            for i, title in enumerate(titles, 1):
                print(f"  {i}. {title[:70]}...")
        
        print("\n" + "-"*70)
        print("Iniciando análisis...")
        print("-"*70)
        
        # Analizar
        analyzer = SimilarityAnalyzer()
        results = analyzer.analyze_texts(abstracts, titles, algorithms)
        
        # Guardar resultados
        analyzer.save_results(results)
        
        # Mostrar resumen
        analyzer.print_summary(results)
        
        # Crear visualización
        try:
            print("\n→ Generando visualización...")
            analyzer.visualize_results(results, 'data/outputs/similarity_visualization.png')
            print("✓ Visualización guardada: data/outputs/similarity_visualization.png")
        except Exception as e:
            print(f"⚠ No se pudo crear visualización: {e}")
        
        # Actualizar estado
        state.similarity_results = results
        
        print("\n" + "="*70)
        print("✓✓✓ REQUERIMIENTO 2 COMPLETADO ✓✓✓")
        print("="*70)
        print("\nResultados guardados en:")
        print("  • data/outputs/similarity_results.json")
        print("  • data/outputs/similarity_visualization.png")
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        logger.error(f"Error en Req 2: {e}")
        import traceback
        traceback.print_exc()

# ========== REQUERIMIENTO 3 ==========
def ejecutar_req3():
    """
    Requerimiento 3: Frecuencia de Palabras + Descubrimiento Semántico
    """
    print("\n" + "="*70)
    print("REQUERIMIENTO 3: ANÁLISIS DE FRECUENCIAS Y NUEVAS PALABRAS")
    print("="*70)

    if not state.has_data():
        print("\nERROR: No hay datos cargados.")
        print("Por favor ejecute el Requerimiento 1 primero.")
        return

    try:
        from R3_Frecuencias import FrequencyAnalyzer

        analyzer = FrequencyAnalyzer()
        predefined_df, precision_df = analyzer.analyze_frequencies(state.df)

        state.frequency_results = {
            "predefined": predefined_df,
            "precision": precision_df
        }

        print("\n" + "="*70)
        print("✓✓✓ REQUERIMIENTO 3 COMPLETADO ✓✓✓")
        print("="*70)
        print("\nResultados guardados en:")
        print("  • data/outputs/frequency_words.csv")
        print("  • data/outputs/new_words_precision.csv")

    except Exception as e:
        print(f"\nERROR: {e}")
        logger.error(f"Error en Req 3: {e}")
        import traceback
        traceback.print_exc()


# ========== REQUERIMIENTO 4 ==========
def ejecutar_req4():
    """
    Requerimiento 4: Clustering Jerárquico
    """
    print("\n" + "="*70)
    print("REQUERIMIENTO 4: CLUSTERING JERÁRQUICO")
    print("="*70)

    if not state.has_data():
        print("\nERROR: No hay datos cargados.")
        print("Por favor ejecute el Requerimiento 1 primero.")
        return

    try:
        from R4_Clustering import ClusteringAnalyzer

        analyzer = ClusteringAnalyzer()
        results = analyzer.analyze_clustering(state.df, n_samples=50)

        print("\n" + "="*70)
        print("✓✓✓ REQUERIMIENTO 4 COMPLETADO ✓✓✓")
        print("="*70)

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()



# ========== REQUERIMIENTO 5 ==========
def ejecutar_req5():
    """
    Requerimiento 5: Visualizaciones y Exportación PDF
    """
    print("\n" + "="*70)
    print("REQUERIMIENTO 5: VISUALIZACIONES Y EXPORTACIÓN")
    print("="*70)

    if not state.has_data():
        print("\nERROR: No hay datos cargados.")
        print("Por favor ejecute el Requerimiento 1 primero.")
        return

    try:
        from R5_Visualizaciones import VisualizationAnalyzer

        analyzer = VisualizationAnalyzer()
        results = analyzer.generate_visualizations(state.df)
        state.visualization_results = results

        print("\n" + "="*70)
        print("✓✓✓ REQUERIMIENTO 5 COMPLETADO ✓✓✓")
        print("="*70)
        print("\nVisualizaciones generadas:")
        for key, path in results.items():
            print(f"  • {key.capitalize()}: {path}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()



# ========== UTILIDADES ==========
def ver_resumen():
    """Muestra resumen de datos cargados"""
    print("\n" + "="*70)
    print("RESUMEN DE DATOS CARGADOS")
    print("="*70)
    
    if not state.has_data():
        print("\n⚠ No hay datos cargados.")
        return
    
    df = state.df
    
    print(f"\nTotal de artículos: {len(df)}")
    
    if 'year' in df.columns:
        print(f"Rango de años: {df['year'].min()} - {df['year'].max()}")
    
    if 'source' in df.columns:
        print("\nDistribución por fuente:")
        print(df['source'].value_counts())
    
    print("\nColumnas disponibles:")
    print(df.columns.tolist())
    
    print("\nPrimeros 3 artículos:")
    print(df[['title', 'year']].head(3).to_string(index=False))


def ver_resultados():
    """Muestra archivos de resultados guardados"""
    print("\n" + "="*70)
    print("ARCHIVOS DE RESULTADOS")
    print("="*70)
    
    output_dir = Path('data/outputs')
    
    if not output_dir.exists():
        print("\n⚠ El directorio data/outputs/ no existe aún.")
        return
    
    files = list(output_dir.glob('*'))
    
    if not files:
        print("\n⚠ No hay resultados guardados aún.")
        return
    
    print("\nArchivos encontrados:")
    print("-"*70)
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  • {f.name:<40} ({size_kb:.1f} KB)")


# ========== MAIN ==========
def main():
    """Función principal con menú interactivo"""
    
    # Crear directorios necesarios
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/outputs').mkdir(parents=True, exist_ok=True)
    
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        try:
            opcion = input("Seleccione una opción: ").strip()
            
            if opcion == '1':
                ejecutar_req1()
                pause()
                
            elif opcion == '2':
                ejecutar_req2()
                pause()
                
            elif opcion == '3':
                ejecutar_req3()
                pause()
                
            elif opcion == '4':
                ejecutar_req4()
                pause()
                
            elif opcion == '5':
                ejecutar_req5()
                pause()
                
            elif opcion == '6':
                ver_resumen()
                pause()
                
            elif opcion == '7':
                ver_resultados()
                pause()
                
            elif opcion == '0':
                print("\n¡Hasta luego!")
                break
                
            else:
                print("\nOpción inválida. Intente nuevamente.")
                pause()
                
        except KeyboardInterrupt:
            print("\n\nPrograma interrumpido por el usuario.")
            sys.exit(0)
        except Exception as e:
            print(f"\nERROR INESPERADO: {str(e)}")
            logger.error(f"Error: {e}")
            pause()


if __name__ == "__main__":
    main()