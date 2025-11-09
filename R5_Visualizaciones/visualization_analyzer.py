"""
Requerimiento 5: Visualizaciones Bibliométricas
VERSIÓN FINAL - Detección precisa de países
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
    print("Plotly no disponible.")

# Importar pycountry (para OpenAlex)
try:
    from pycountry import countries as pycountries
    PYCOUNTRY_AVAILABLE = True
except ImportError:
    PYCOUNTRY_AVAILABLE = False

# Importar requests (para OpenAlex)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


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
        
        # Cache para OpenAlex
        self.country_cache = {}
        
        # Contador de métodos de detección
        self.detection_stats = {
            'affiliations': 0,
            'venue': 0,
            'openalex': 0,
            'unknown': 0
        }

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
    # 1. MAPA DE CALOR GEOGRÁFICO
    # ============================================

    def _generate_heatmap(self, df):
        """Genera mapa geográfico mundial basado en país del primer autor."""
        
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly no está instalado.")

        # 1. Detectar países (método mejorado)
        print("Detectando países de autores...")
        df_with_country = self._detect_country_comprehensive(df)
        
        # 2. Contar artículos por país
        country_data = self._count_articles_by_country(df_with_country)

        if len(country_data) == 0:
            raise ValueError(
                "No se encontraron países válidos en los datos.\n"
                "Verifica que el DataFrame tenga columnas: 'affiliations', 'venue' o 'doi'"
            )

        print(f"Países detectados: {len(country_data)}")
        print(f"Top 3: {', '.join(country_data.head(3)['country'].tolist())}")
        
        # Mostrar estadísticas de detección
        print(f"\nEstadísticas de detección:")
        print(f"      • Desde affiliations: {self.detection_stats['affiliations']}")
        print(f"      • Desde venue: {self.detection_stats['venue']}")
        print(f"      • Desde OpenAlex: {self.detection_stats['openalex']}")
        print(f"      • No detectados: {self.detection_stats['unknown']}")

        # 3. Crear mapa
        fig = px.choropleth(
            country_data,
            locations="country",
            locationmode="country names",
            color="count",
            hover_name="country",
            hover_data={"count": True, "percentage": ':.2f'},
            color_continuous_scale="YlOrRd",
            labels={'count': 'Artículos', 'percentage': 'Porcentaje (%)'},
            title="<b>Distribución Geográfica por País del Primer Autor</b>",
        )

        fig.update_layout(
            geo=dict(
                showframe=False, 
                showcoastlines=True, 
                coastlinecolor="Gray",
                projection_type="natural earth",
                bgcolor='rgba(240,240,240,0.5)'
            ),
            title={
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            margin=dict(l=0, r=0, t=50, b=0),
            height=600,
            coloraxis_colorbar=dict(
                title="Número de<br>Artículos",
                thickness=20,
                len=0.7
            )
        )

        # 4. Guardar SOLO PNG (sin HTML)
        path = self.output_dir / "heatmap.png"
        fig.write_image(str(path), width=1200, height=600, scale=2)
        
        return path

    # ============================================
    # DETECCIÓN MEJORADA DE PAÍSES
    # ============================================

    def _detect_country_comprehensive(self, df):
        """
        Detecta país del primer autor usando método exhaustivo:
        
        1. Columna 'country' directa
        2. Parsing profundo de 'affiliations' (regex mejorado)
        3. OpenAlex API con todos los DOIs (con rate limiting)
        4. Inferencia desde 'venue'
        5. Marca como 'Unknown' si no se detecta
        
        NO asigna países aleatorios - solo datos reales.
        """
        
        df = df.copy()
        df['first_author_country'] = None
        
        total = len(df)
        
        # ========================================
        # MÉTODO 1: Columna 'country' directa
        # ========================================
        if 'country' in df.columns:
            print("Método 1: Columna 'country' directa...")
            mask = df['country'].notna()
            df.loc[mask, 'first_author_country'] = df.loc[mask, 'country']
            detected = df['first_author_country'].notna().sum()
            print(f"      → Detectados: {detected}/{total}")
        
        # ========================================
        # MÉTODO 2: Parsing de 'affiliations'
        # ========================================
        affiliation_cols = ['affiliations', 'affiliation', 'author_affiliation', 'author_affiliations']
        
        for col in affiliation_cols:
            if col in df.columns:
                print(f"Método 2: Parsing de '{col}'...")
                mask = df['first_author_country'].isna() & df[col].notna()
                
                if mask.any():
                    countries_detected = df.loc[mask, col].apply(self._parse_country_advanced)
                    valid_mask = mask & countries_detected.notna()
                    df.loc[valid_mask, 'first_author_country'] = countries_detected[valid_mask]
                    self.detection_stats['affiliations'] += valid_mask.sum()
                    
                    current_detected = df['first_author_country'].notna().sum()
                    print(f"      → Detectados: {current_detected}/{total}")
        
        # ========================================
        # MÉTODO 3: OpenAlex API (TODOS los DOIs)
        # ========================================
        if REQUESTS_AVAILABLE and 'doi' in df.columns:
            print("Método 3: Consultando OpenAlex API...")
            
            missing_mask = df['first_author_country'].isna() & df['doi'].notna()
            missing_dois = df[missing_mask]
            
            if len(missing_dois) > 0:
                print(f"      → Consultando {len(missing_dois)} DOIs...")
                
                # Procesar TODOS los DOIs (con progress)
                for i, (idx, row) in enumerate(missing_dois.iterrows(), 1):
                    if i % 10 == 0:
                        print(f"         Progreso: {i}/{len(missing_dois)}")
                    
                    country = self._get_country_from_openalex(row.get('doi'))
                    if country:
                        df.at[idx, 'first_author_country'] = country
                        self.detection_stats['openalex'] += 1
                    
                    # Rate limiting: 1 request cada 0.5 segundos
                    time.sleep(0.5)
                
                current_detected = df['first_author_country'].notna().sum()
                print(f"      → Detectados: {current_detected}/{total}")
        
        # ========================================
        # MÉTODO 4: Inferencia desde 'venue'
        # ========================================
        venue_cols = ['venue', 'journal', 'publication_venue']
        
        for col in venue_cols:
            if col in df.columns:
                print(f"Método 4: Inferencia desde '{col}'...")
                mask = df['first_author_country'].isna() & df[col].notna()
                
                if mask.any():
                    countries_detected = df.loc[mask, col].apply(self._infer_country_from_venue)
                    valid_mask = mask & countries_detected.notna()
                    df.loc[valid_mask, 'first_author_country'] = countries_detected[valid_mask]
                    self.detection_stats['venue'] += valid_mask.sum()
                    
                    current_detected = df['first_author_country'].notna().sum()
                    print(f"      → Detectados: {current_detected}/{total}")
        
        # ========================================
        # MÉTODO 5: Marcar Unknown (NO asignar aleatorios)
        # ========================================
        unknown_mask = df['first_author_country'].isna()
        unknown_count = unknown_mask.sum()
        
        if unknown_count > 0:
            df.loc[unknown_mask, 'first_author_country'] = 'Unknown'
            self.detection_stats['unknown'] = unknown_count
            print(f"\nAdvertencia: {unknown_count} registros sin país detectado")
            print(f"Estos se marcarán como 'Unknown' y NO aparecerán en el mapa")
        
        return df

    def _parse_country_advanced(self, text):
        """
        Parsing avanzado de países desde texto de afiliación.
        Usa múltiples patrones y regex.
        """
        
        if pd.isna(text) or str(text).strip() == '':
            return None
        
        text = str(text).lower()
        
        # Diccionario extendido de países con múltiples patrones
        country_patterns = {
            'United States': [
                r'\busa\b', r'\bu\.s\.a\b', r'\bu\.s\.\b', r'\bunited states\b',
                r'\bamerica\b', r'\bus\b(?!$)',  # 'us' no al final
                r'\bcalifornia\b', r'\bmassachusetts\b', r'\btexas\b', r'\bnew york\b',
                r'\bboston\b', r'\blos angeles\b', r'\bchicago\b', r'\bseattle\b',
                r'\bmit\b', r'\bstanford\b', r'\bharvard\b', r'\bberkeley\b'
            ],
            'United Kingdom': [
                r'\buk\b', r'\bu\.k\.\b', r'\bunited kingdom\b', r'\bbritain\b',
                r'\bengland\b', r'\bscotland\b', r'\bwales\b',
                r'\blondon\b', r'\boxford\b', r'\bcambridge\b', r'\bedinburgh\b',
                r'\bmanchester\b', r'\bbirmingham\b'
            ],
            'China': [
                r'\bchina\b', r'\bprc\b', r'\bbeijing\b', r'\bshanghai\b',
                r'\bguangzhou\b', r'\bshenzhen\b', r'\bhong kong\b', r'\btsinghua\b',
                r'\bpeking\b', r'\btaiwan\b'
            ],
            'India': [
                r'\bindia\b', r'\bdelhi\b', r'\bmumbai\b', r'\bbangalore\b',
                r'\bhyd erabad\b', r'\bchennai\b', r'\bkolkata\b', r'\bpune\b'
            ],
            'Germany': [
                r'\bgermany\b', r'\bdeutschland\b', r'\bberlin\b', r'\bmunich\b',
                r'\bfrankfurt\b', r'\bhamburg\b', r'\bcologne\b', r'\bstuttgart\b',
                r'\bmax planck\b'
            ],
            'France': [
                r'\bfrance\b', r'\bparis\b', r'\blyon\b', r'\bmarseille\b',
                r'\btoulouse\b', r'\bnice\b', r'\bstrasbourg\b'
            ],
            'Canada': [
                r'\bcanada\b', r'\btoronto\b', r'\bmontreal\b', r'\bvancouver\b',
                r'\bottawa\b', r'\bcalgary\b', r'\bedmonton\b'
            ],
            'Australia': [
                r'\baustralia\b', r'\bsydney\b', r'\bmelbourne\b', r'\bbrisbane\b',
                r'\bperth\b', r'\badelaide\b', r'\bcanberra\b'
            ],
            'Japan': [
                r'\bjapan\b', r'\btokyo\b', r'\bosaka\b', r'\bkyoto\b',
                r'\byokohama\b', r'\bnagoya\b', r'\bsapporo\b'
            ],
            'Spain': [
                r'\bspain\b', r'\bespaña\b', r'\bmadrid\b', r'\bbarcelona\b',
                r'\bvalencia\b', r'\bseville\b', r'\bbilbao\b'
            ],
            'Italy': [
                r'\bitaly\b', r'\bitalia\b', r'\brome\b', r'\bmilan\b',
                r'\bnaples\b', r'\bturin\b', r'\bflorence\b', r'\bvenice\b'
            ],
            'Brazil': [
                r'\bbrazil\b', r'\bbrasil\b', r'\bsão paulo\b', r'\brio\b',
                r'\bbrasilia\b', r'\bbelo horizonte\b'
            ],
            'South Korea': [
                r'\bsouth korea\b', r'\bkorea\b', r'\bseoul\b', r'\bbusan\b',
                r'\bincheon\b', r'\bdaegu\b'
            ],
            'Mexico': [
                r'\bmexico\b', r'\bméxico\b', r'\bciudad de mexico\b',
                r'\bguadalajara\b', r'\bmonterrey\b'
            ],
            'Colombia': [
                r'\bcolombia\b', r'\bbogotá\b', r'\bbogota\b', r'\bmedellín\b',
                r'\bmedellin\b', r'\bcali\b', r'\bbarranquilla\b', r'\bcartagena\b'
            ],
            'Argentina': [
                r'\bargentina\b', r'\bbuenos aires\b', r'\bcórdoba\b',
                r'\brosario\b', r'\bmendoza\b'
            ],
            'Netherlands': [
                r'\bnetherlands\b', r'\bholland\b', r'\bamsterdam\b',
                r'\brotterdam\b', r'\bthe hague\b', r'\butrecht\b'
            ],
            'Switzerland': [
                r'\bswitzerland\b', r'\bsuisse\b', r'\bzurich\b',
                r'\bgeneva\b', r'\bbern\b', r'\blausanne\b', r'\bethz\b'
            ],
            'Sweden': [
                r'\bsweden\b', r'\bstockholm\b', r'\bgothenburg\b',
                r'\bmalmö\b', r'\bupsala\b'
            ],
            'Singapore': [
                r'\bsingapore\b', r'\bnus\b', r'\bntu\b'
            ],
            'Russia': [
                r'\brussia\b', r'\bmoscow\b', r'\bst petersburg\b',
                r'\bnovosibirsk\b'
            ],
            'Poland': [
                r'\bpoland\b', r'\bwarsaw\b', r'\bkrakow\b', r'\bwroclaw\b'
            ],
            'Belgium': [
                r'\bbelgium\b', r'\bbrussels\b', r'\bantwerp\b', r'\bghent\b'
            ],
            'Austria': [
                r'\baustria\b', r'\bvienna\b', r'\bsalzburg\b', r'\binnsbruck\b'
            ],
            'Denmark': [
                r'\bdenmark\b', r'\bcopenhagen\b', r'\baarhus\b'
            ],
            'Norway': [
                r'\bnorway\b', r'\boslo\b', r'\bbergen\b'
            ],
            'Finland': [
                r'\bfinland\b', r'\bhelsinki\b', r'\bespoo\b'
            ],
            'Ireland': [
                r'\bireland\b', r'\bdublin\b', r'\bcork\b'
            ],
            'Portugal': [
                r'\bportugal\b', r'\blisbon\b', r'\bporto\b'
            ],
            'Greece': [
                r'\bgreece\b', r'\bathens\b', r'\bthessaloniki\b'
            ],
            'Turkey': [
                r'\bturkey\b', r'\bistanbul\b', r'\bankara\b'
            ],
            'Israel': [
                r'\bisrael\b', r'\btel aviv\b', r'\bjerusalem\b', r'\bhaifa\b'
            ],
            'South Africa': [
                r'\bsouth africa\b', r'\bcape town\b', r'\bjohannesburg\b'
            ],
            'Chile': [
                r'\bchile\b', r'\bsantiago\b', r'\bvalparaiso\b'
            ],
            'Peru': [
                r'\bperu\b', r'\bperú\b', r'\blima\b', r'\bcusco\b'
            ],
            'Ecuador': [
                r'\becuador\b', r'\bquito\b', r'\bguayaquil\b'
            ],
            'Venezuela': [
                r'\bvenezuela\b', r'\bcaracas\b', r'\bmaracaibo\b'
            ]
        }
        
        # Buscar usando regex (más preciso)
        for country, patterns in country_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return country
        
        return None

    def _infer_country_from_venue(self, venue):
        """Infiere país desde nombre de revista/conferencia."""
        
        if pd.isna(venue) or venue == '':
            return None
        
        venue = str(venue).lower()
        
        # Patrones en nombres de venues
        patterns = {
            'United States': ['american', 'ieee usa', 'acm usa'],
            'United Kingdom': ['british', 'london', 'oxford journal', 'cambridge press'],
            'China': ['chinese', 'china journal'],
            'India': ['indian'],
            'Germany': ['german', 'springer'],
            'France': ['french', 'elsevier france'],
            'Spain': ['spanish', 'español'],
            'Colombia': ['colombian', 'universidad colombia']
        }
        
        for country, keywords in patterns.items():
            if any(kw in venue for kw in keywords):
                return country
        
        return None

    def _get_country_from_openalex(self, doi):
        """Obtiene país del primer autor usando OpenAlex API."""
        
        if not REQUESTS_AVAILABLE or not PYCOUNTRY_AVAILABLE:
            return None
        
        if not doi or not isinstance(doi, str):
            return None
        
        # Cache
        if doi in self.country_cache:
            return self.country_cache[doi]
        
        # Limpiar DOI
        doi = str(doi).strip().replace("https://doi.org/", "")
        if not doi:
            return None
        
        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        
        try:
            response = requests.get(url, timeout=15)
            
            if response.status_code != 200:
                self.country_cache[doi] = None
                return None
            
            data = response.json()
            authorships = data.get("authorships", [])
            
            if not authorships:
                self.country_cache[doi] = None
                return None
            
            # Primer autor
            first_author = authorships[0]
            institutions = first_author.get("institutions", [])
            
            if institutions:
                country_code = institutions[0].get("country_code")
                if country_code:
                    try:
                        country_name = pycountries.get(alpha_2=country_code.upper()).name
                        self.country_cache[doi] = country_name
                        return country_name
                    except:
                        pass
            
            self.country_cache[doi] = None
            return None
        
        except Exception:
            self.country_cache[doi] = None
            return None

    def _count_articles_by_country(self, df):
        """Cuenta artículos por país"""
        
        if 'first_author_country' not in df.columns:
            raise ValueError("DataFrame no tiene columna 'first_author_country'")
        
        # Contar (excluyendo 'Unknown')
        counts = df[df['first_author_country'] != 'Unknown']['first_author_country'].value_counts()
        total = len(df[df['first_author_country'] != 'Unknown'])
        
        if total == 0:
            raise ValueError("No se detectaron países válidos. Todos son 'Unknown'.")
        
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
        """Genera nube de palabras basada en abstracts Y keywords."""
        
        texts = []
        
        if 'abstract' in df.columns:
            abstracts = df['abstract'].dropna().astype(str)
            texts.extend(abstracts.tolist())
        
        if 'keywords' in df.columns:
            keywords = df['keywords'].dropna().astype(str)
            texts.extend(keywords.tolist())
        
        if not texts:
            raise ValueError("No hay texto disponible para generar nube de palabras")
        
        full_text = " ".join(texts)
        
        if len(full_text) < 50:
            raise ValueError("No hay suficiente texto en abstracts/keywords")
        
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
        """Genera línea temporal de publicaciones por año."""
        
        if 'year' not in df.columns:
            raise ValueError("El DataFrame no contiene columna 'year'")
        
        df_filtered = df[df['year'] > 0].copy()
        
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
        timeline_data = pd.crosstab(df['year'], df[source_col])
        
        plt.figure(figsize=(12, 6))
        
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
        """Exporta visualizaciones a PDF profesional"""
        
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
                
                # PORTADA
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                ax.text(0.5, 0.7, 'Reporte de Visualizaciones\nAnálisis Bibliométrico',
                        ha='center', va='center', fontsize=24, fontweight='bold',
                        color='#2C3E50')
                ax.text(0.5, 0.6, 'Proyecto: Análisis de Algoritmos',
                        ha='center', va='center', fontsize=14, color='#34495E')
                
                date_str = datetime.now().strftime('%d/%m/%Y %H:%M')
                ax.text(0.5, 0.5, f'Generado el: {date_str}',
                        ha='center', va='center', fontsize=12, color='#7F8C8D')
                
                description = (
                    'Este reporte contiene las visualizaciones del Requerimiento 5:\n\n'
                    '• Mapa de calor geográfico (distribución por país)\n'
                    '• Nube de palabras (abstracts + keywords)\n'
                    '• Línea temporal (publicaciones por año)'
                )
                ax.text(0.5, 0.3, description, ha='center', va='center',
                        fontsize=11, color='#34495E')
                
                ax.text(0.5, 0.1, 'Universidad del Quindío\nIngeniería de Sistemas y Computación',
                        ha='center', va='center', fontsize=9, color='#95A5A6')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                # VISUALIZACIONES
                for i, img_path in enumerate(image_paths):
                    if not img_path.exists():
                        print(f"Imagen no encontrada: {img_path}")
                        continue
                    
                    try:
                        img = PILImage.open(img_path)
                        img_array = np.array(img)
                        
                        fig = plt.figure(figsize=(8.5, 11))
                        fig.text(0.5, 0.95, titles[i], ha='center',
                                fontsize=14, fontweight='bold', color='#2C3E50')
                        fig.text(0.5, 0.91, descriptions[i], ha='center',
                                fontsize=10, color='#7F8C8D')
                        
                        ax = fig.add_axes([0.1, 0.15, 0.8, 0.7])
                        ax.imshow(img_array)
                        ax.axis('off')
                        
                        fig.text(0.5, 0.05, f'Página {i + 2} de {len(image_paths) + 1}',
                                ha='center', fontsize=9, color='#95A5A6')
                        
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                    
                    except Exception as e:
                        print(f"Error al procesar {img_path.name}: {e}")
                        continue
                
                # METADATA
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