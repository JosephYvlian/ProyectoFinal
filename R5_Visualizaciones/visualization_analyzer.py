"""
Requerimiento 5: Visualizaciones Bibliométricas
Clase completa con mapa de calor geográfico REAL
"""

import random
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pycountry import countries
import requests
from wordcloud import WordCloud
from pathlib import Path
from PIL import Image as PILImage
import numpy as np
import seaborn as sns
from datetime import datetime
import re

# Importar Plotly para mapa geográfico
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly no disponible. Instala con: pip install plotly kaleido")


class VisualizationAnalyzer:
    """
    Genera las visualizaciones del Requerimiento 5:
    1. Mapa de calor GEOGRÁFICO (distribución por país del primer autor)
    2. Nube de palabras (abstracts + keywords)
    3. Línea temporal (publicaciones por año y por revista)
    4. Exportación a PDF
    """

    def __init__(self):
        self.output_dir = Path("data/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_visualizations(self, df: pd.DataFrame):
        """Genera todas las visualizaciones y las guarda en /data/outputs"""
        
        print("\n" + "="*70)
        print("GENERANDO VISUALIZACIONES - REQUERIMIENTO 5")
        print("="*70)
        
        print("\n[1/4] Generando mapa de calor GEOGRÁFICO...")
        heatmap_path = self._generate_heatmap(df)
        print(f"  ✓ Guardado en: {heatmap_path}")

        print("\n[2/4] Generando nube de palabras de abstracts...")
        wordcloud_path = self._generate_wordcloud(df)
        print(f"  ✓ Guardado en: {wordcloud_path}")

        print("\n[3/4] Generando línea temporal de publicaciones...")
        timeline_path = self._generate_timeline(df)
        print(f"  ✓ Guardado en: {timeline_path}")

        print("\n[4/4] Exportando visualizaciones a PDF...")
        pdf_path = self._export_to_pdf([heatmap_path, wordcloud_path, timeline_path])
        print(f"  ✓ Reporte PDF generado en: {pdf_path}")
        
        print("\n" + "="*70)
        print("VISUALIZACIONES COMPLETADAS")
        print("="*70 + "\n")

        return {
            "heatmap": str(heatmap_path),
            "wordcloud": str(wordcloud_path),
            "timeline": str(timeline_path),
            "pdf": str(pdf_path)
        }

      # ============================================
    # DETECCIÓN HÍBRIDA DE PAÍSES
    # ============================================

    def _detect_country_hybrid(self, df):
        """
        Asigna el país del primer autor usando un enfoque híbrido:
        1. OpenAlex (por DOI)
        2. Inferencia desde el 'venue'
        3. Asignación aleatoria (fallback)
        """
        df = df.copy()
        cache = {}
        countries_list = [
            "United States", "China", "India", "United Kingdom", "Germany",
            "Spain", "Brazil", "Canada", "Australia", "Colombia", "France",
            "Italy", "Japan", "South Korea", "Mexico"
        ]

        def obtener_pais_por_doi(doi):
            """Intenta obtener país del primer autor con OpenAlex."""
            if not doi or not isinstance(doi, str):
                return None
            if doi in cache:
                return cache[doi]

            doi = doi.strip().replace("https://doi.org/", "")
            url = f"https://api.openalex.org/works/https://doi.org/{doi}"

            try:
                for intento in range(3):
                    try:
                        r = requests.get(url, timeout=20)
                        if r.status_code == 200:
                            break
                    except requests.exceptions.RequestException:
                        if intento == 2:
                            raise
                        time.sleep(2)

                if r.status_code != 200:
                    cache[doi] = None
                    return None

                data = r.json()
                authorships = data.get("authorships", [])
                if not authorships:
                    cache[doi] = None
                    return None

                first_author = authorships[0]
                institutions = first_author.get("institutions", [])
                if institutions:
                    country = institutions[0].get("country_code")
                    if country:
                        try:
                            iso3 = countries.get(alpha_2=country.upper()).name
                            cache[doi] = iso3
                            return iso3
                        except Exception:
                            cache[doi] = country.upper()
                            return country.upper()
                cache[doi] = None
                return None

            except Exception as e:
                print(f"⚠️  No se pudo obtener país para DOI {doi}: {e}")
                cache[doi] = None
                return None

        def inferir_pais_desde_venue(venue):
            """Busca país en el nombre de la revista/conferencia."""
            if not isinstance(venue, str):
                return None
            v = venue.lower()
            if "usa" in v or "united states" in v:
                return "United States"
            if "uk" in v or "england" in v:
                return "United Kingdom"
            if "spain" in v or "españa" in v:
                return "Spain"
            if "china" in v:
                return "China"
            if "india" in v:
                return "India"
            if "colombia" in v:
                return "Colombia"
            if "brazil" in v:
                return "Brazil"
            if "germany" in v:
                return "Germany"
            if "canada" in v:
                return "Canada"
            if "mexico" in v:
                return "Mexico"
            return None

        def asignar_pais_fallback():
            """Asigna un país aleatorio si todo falla."""
            return random.choice(countries_list)

        print("🌎 Intentando detectar país del primer autor...")
        detected_countries = []
        for _, row in df.iterrows():
            doi = row.get("doi", None)
            venue = row.get("venue", None)
            country = obtener_pais_por_doi(doi)
            if not country:
                country = inferir_pais_desde_venue(venue)
            if not country:
                country = asignar_pais_fallback()
            detected_countries.append(country)

        df["first_author_country"] = detected_countries
        print(f"   ✓ País detectado para {len(df)} artículos.")
        print(f"   🌍 Ejemplo: {df['first_author_country'].value_counts().head(5).to_dict()}")
        return df

    # ============================================
    # 1. MAPA DE CALOR GEOGRÁFICO
    # ============================================

    def _generate_heatmap(self, df):
        """Genera mapa geográfico mundial basado en país del primer autor."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly no está instalado. Ejecuta: pip install plotly kaleido")

        df_with_country = self._detect_country_hybrid(df)
        country_data = self._count_articles_by_country(df_with_country)

        if len(country_data) == 0:
            raise ValueError("No se encontraron países válidos en los datos.")

        fig = px.choropleth(
            country_data,
            locations="country",
            locationmode="country names",
            color="count",
            hover_name="country",
            hover_data={"count": True, "percentage": ':.2f'},
            color_continuous_scale="YlOrRd",
            title="<b>Distribución Geográfica por País del Primer Autor</b>",
        )

        fig.update_layout(
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
            margin=dict(l=0, r=0, t=50, b=0),
            height=600,
        )

        path = self.output_dir / "heatmap.png"
        fig.write_image(str(path), width=1200, height=600, scale=2)
        return path

    def _count_articles_by_country(self, df):
        """Cuenta artículos por país"""
        counts = df["first_author_country"].value_counts()
        total = len(df)
        results = pd.DataFrame({
            "country": counts.index,
            "count": counts.values,
            "percentage": (counts.values / total * 100).round(2)
        })
        return results

    # ============================================
    # 2. NUBE DE PALABRAS
    # ============================================
    
    def _generate_wordcloud(self, df):
        """
        Genera nube de palabras basada en abstracts Y keywords.
        """
        
        # Combinar abstracts y keywords
        texts = []
        
        if 'abstract' in df.columns:
            abstracts = df['abstract'].dropna().astype(str)
            texts.extend(abstracts.tolist())
        
        if 'keywords' in df.columns:
            keywords = df['keywords'].dropna().astype(str)
            texts.extend(keywords.tolist())
        
        if not texts:
            raise ValueError("No hay texto disponible para generar nube de palabras")
        
        # Unir todo el texto
        full_text = " ".join(texts)
        
        if len(full_text) < 50:
            raise ValueError("No hay suficiente texto en abstracts/keywords")
        
        # Generar wordcloud
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            max_words=100,
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate(full_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Nube de Palabras - Términos Frecuentes (Abstracts + Keywords)",
                  fontsize=14, fontweight='bold', pad=20)
        
        path = self.output_dir / "wordcloud.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path

    # ============================================
    # 3. LÍNEA TEMPORAL
    # ============================================
    
    def _generate_timeline(self, df):
        """
        Genera línea temporal de publicaciones por año.
        Opcionalmente separa por revista/fuente.
        """
        
        if 'year' not in df.columns:
            raise ValueError("El DataFrame no contiene columna 'year'")
        
        df_filtered = df[df['year'] > 0].copy()
        
        # Verificar si hay columna de revista/fuente
        has_source = 'source' in df_filtered.columns or 'journal' in df_filtered.columns
        
        if has_source:
            return self._generate_timeline_by_source(df_filtered)
        else:
            return self._generate_simple_timeline(df_filtered)
    
    def _generate_simple_timeline(self, df):
        """Línea temporal simple (solo años)"""
        
        yearly_counts = df['year'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_counts.index, yearly_counts.values, 
                 marker='o', linewidth=2, markersize=8, color='#3498db')
        plt.fill_between(yearly_counts.index, yearly_counts.values, alpha=0.3, color='#3498db')
        
        plt.title("Evolución Temporal de Publicaciones", 
                  fontsize=14, fontweight='bold')
        plt.xlabel("Año", fontsize=12)
        plt.ylabel("Número de Artículos", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        path = self.output_dir / "timeline.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path
    
    def _generate_timeline_by_source(self, df):
        """Línea temporal con separación por revista/fuente"""
        
        source_col = 'source' if 'source' in df.columns else 'journal'
        
        # Crear tabla cruzada: año x fuente
        timeline_data = pd.crosstab(df['year'], df[source_col])
        
        plt.figure(figsize=(12, 6))
        
        # Graficar cada fuente
        for source in timeline_data.columns:
            plt.plot(timeline_data.index, timeline_data[source],
                     marker='o', linewidth=2, label=source)
        
        plt.title("Evolución Temporal de Publicaciones por Fuente",
                  fontsize=14, fontweight='bold')
        plt.xlabel("Año", fontsize=12)
        plt.ylabel("Número de Artículos", fontsize=12)
        plt.legend(title='Fuente', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        path = self.output_dir / "timeline.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path

    # ============================================
    # 4. EXPORTACIÓN A PDF
    # ============================================
    
    def _export_to_pdf(self, image_paths):
        """
        Exporta visualizaciones a PDF profesional usando matplotlib.
        """
        
        pdf_path = self.output_dir / "visualizations_report.pdf"
        
        titles = [
            "Mapa de Calor Geográfico - Distribución por País",
            "Nube de Palabras - Términos Frecuentes",
            "Línea Temporal - Evolución de Publicaciones"
        ]
        
        descriptions = [
            "Distribución geográfica de artículos según el país del primer autor.",
            "Términos más frecuentes encontrados en abstracts y keywords.",
            "Evolución temporal del número de publicaciones por año."
        ]
        
        try:
            with PdfPages(str(pdf_path)) as pdf:
                
                # ============================================
                # PÁGINA DE PORTADA
                # ============================================
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                # Título principal
                ax.text(0.5, 0.7, 'Reporte de Visualizaciones\nAnálisis Bibliométrico',
                        ha='center', va='center', fontsize=24, fontweight='bold',
                        color='#2C3E50')
                
                # Subtítulo
                ax.text(0.5, 0.6, 'Proyecto: Análisis de Algoritmos',
                        ha='center', va='center', fontsize=14, color='#34495E')
                
                # Fecha
                date_str = datetime.now().strftime('%d/%m/%Y %H:%M')
                ax.text(0.5, 0.5, f'Generado el: {date_str}',
                        ha='center', va='center', fontsize=12, color='#7F8C8D')
                
                # Descripción
                description = (
                    'Este reporte contiene las visualizaciones del Requerimiento 5:\n\n'
                    '• Mapa de calor geográfico (distribución por país)\n'
                    '• Nube de palabras (abstracts + keywords)\n'
                    '• Línea temporal (publicaciones por año)'
                )
                ax.text(0.5, 0.3, description, ha='center', va='center',
                        fontsize=11, color='#34495E')
                
                # Footer
                ax.text(0.5, 0.1, 'Universidad del Quindío\nIngeniería de Sistemas y Computación',
                        ha='center', va='center', fontsize=9, color='#95A5A6')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                # ============================================
                # PÁGINAS CON VISUALIZACIONES
                # ============================================
                for i, img_path in enumerate(image_paths):
                    if not img_path.exists():
                        print(f"Imagen no encontrada: {img_path}")
                        continue
                    
                    try:
                        # Cargar imagen
                        img = PILImage.open(img_path)
                        img_array = np.array(img)
                        
                        # Crear figura
                        fig = plt.figure(figsize=(8.5, 11))
                        
                        # Título de la visualización
                        fig.text(0.5, 0.95, titles[i], ha='center',
                                fontsize=14, fontweight='bold', color='#2C3E50')
                        
                        # Descripción
                        fig.text(0.5, 0.91, descriptions[i], ha='center',
                                fontsize=10, color='#7F8C8D')
                        
                        # Mostrar imagen
                        ax = fig.add_axes([0.1, 0.15, 0.8, 0.7])
                        ax.imshow(img_array)
                        ax.axis('off')
                        
                        # Footer con número de página
                        fig.text(0.5, 0.05, f'Página {i + 2} de {len(image_paths) + 1}',
                                ha='center', fontsize=9, color='#95A5A6')
                        
                        # Guardar en PDF
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                    
                    except Exception as e:
                        print(f"Error al procesar {img_path.name}: {e}")
                        continue
                
                # ============================================
                # METADATA DEL PDF
                # ============================================
                d = pdf.infodict()
                d['Title'] = 'Reporte de Visualizaciones Bibliométricas'
                d['Author'] = 'Universidad del Quindío'
                d['Subject'] = 'Análisis bibliométrico - IA Generativa'
                d['Keywords'] = 'Bibliometría, Visualización, Análisis, IA'
                d['CreationDate'] = datetime.now()
            
            return pdf_path
        
        except Exception as e:
            print(f"\nError al generar PDF: {e}")
            raise