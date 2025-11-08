import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import seaborn as sns

class VisualizationAnalyzer:
    """
    Genera las visualizaciones del Requerimiento 5:
    - Mapa de calor (año vs fuente)
    - Nube de palabras (abstracts)
    - Línea temporal (cantidad de publicaciones por año)
    - Exportación a PDF
    """

    def __init__(self):
        self.output_dir = Path("data/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_visualizations(self, df: pd.DataFrame):
        """Genera todas las visualizaciones y las guarda en /data/outputs"""
        print("\n[1/4] Generando mapa de calor por año y fuente...")
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

        return {
            "heatmap": str(heatmap_path),
            "wordcloud": str(wordcloud_path),
            "timeline": str(timeline_path),
            "pdf": str(pdf_path)
        }

    # ------------------------------------------------------------------
    def _generate_heatmap(self, df):
        """Genera mapa de calor: cantidad de artículos por año y fuente."""
        if 'year' not in df.columns or 'source' not in df.columns:
            raise ValueError("El DataFrame no contiene columnas 'year' y 'source'.")

        pivot = df.pivot_table(index='year', columns='source', values='title', aggfunc='count', fill_value=0)
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")
        plt.title("Distribución de artículos por año y fuente")
        plt.ylabel("Año")
        plt.xlabel("Fuente")

        path = self.output_dir / "heatmap.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    # ------------------------------------------------------------------
    def _generate_wordcloud(self, df):
        """Genera nube de palabras basada en los abstracts."""
        text = " ".join(df['abstract'].dropna().astype(str))
        if len(text) < 50:
            raise ValueError("No hay suficiente texto en los abstracts.")

        wordcloud = WordCloud(width=1000, height=600, background_color="white", max_words=100).generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Nube de Palabras - Abstracts")

        path = self.output_dir / "wordcloud.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    # ------------------------------------------------------------------
    def _generate_timeline(self, df):
        """Genera línea temporal de número de publicaciones por año."""
        if 'year' not in df.columns:
            raise ValueError("El DataFrame no contiene columna 'year'.")

        df = df[df['year'] > 0]
        yearly_counts = df['year'].value_counts().sort_index()

        plt.figure(figsize=(10, 5))
        plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2)
        plt.title("Evolución de publicaciones por año")
        plt.xlabel("Año")
        plt.ylabel("Número de artículos")
        plt.grid(True)

        path = self.output_dir / "timeline.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    # ------------------------------------------------------------------
    def _export_to_pdf(self, image_paths):
        """Exporta las imágenes a un PDF."""
        pdf_path = self.output_dir / "visualizations_report.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter

        for img_path in image_paths:
            c.drawImage(str(img_path), 50, 150, width=500, height=400)
            c.showPage()

        c.save()
        return pdf_path
