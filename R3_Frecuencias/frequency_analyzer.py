"""
Requerimiento 3: Análisis de Frecuencia de Palabras y Descubrimiento Semántico
Implementación robusta: evita importaciones pesadas a nivel de módulo y maneja errores
de dependencias para que la importación no falle en el main.
"""

from pathlib import Path
from collections import Counter
import re
import logging
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)
# No importamos SentenceTransformer ni util aquí (lo haremos de forma "lazy")

class FrequencyAnalyzer:
    """
    Analiza frecuencias de términos y genera nuevas palabras asociadas
    usando TF-IDF + embeddings semánticos (Sentence-BERT) si está disponible.

    Uso:
        analyzer = FrequencyAnalyzer()
        predefined_df, precision_df = analyzer.analyze_frequencies(df)
    """

    def __init__(self, sbert_model_name: str = "all-MiniLM-L6-v2"):
        # Palabras esperadas según el requerimiento
        self.predefined_words = [
            "Generative models", "Prompting", "Machine learning", "Multimodality",
            "Fine-tuning", "Training data", "Algorithmic bias", "Explainability",
            "Transparency", "Ethics", "Privacy", "Personalization",
            "Human-AI interaction", "AI literacy", "Co-creation"
        ]
        # Nombre del modelo SBERT (puede cambiarse)
        self.sbert_model_name = sbert_model_name

        # Variable para la instancia del modelo (lazy load)
        self._sbert_model = None
        self._sbert_util = None

    # --------------------------
    # UTILIDADES
    # --------------------------
    def _clean_text(self, text: str) -> str:
        if text is None:
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # --------------------------
    # CARGA LA LIBRERÍA SBERT (LA CARGA SOLO SI SE USA)
    # --------------------------
    def _ensure_sbert(self):
        """Carga sentence-transformers cuando sea necesario (lazy)."""
        if self._sbert_model is None:
            try:
                from sentence_transformers import SentenceTransformer, util
                logger.info(f"Cargando modelo SBERT: {self.sbert_model_name} ...")
                self._sbert_model = SentenceTransformer(self.sbert_model_name)
                self._sbert_util = util
                logger.info("Modelo SBERT cargado correctamente.")
            except Exception as e:
                # No lanzar excepción aquí para no romper la importación del módulo
                logger.error("No se pudo cargar 'sentence-transformers'. ")
                logger.debug(f"Detalle: {e}")
                self._sbert_model = None
                self._sbert_util = None

    # --------------------------
    # FRECUENCIA DE PALABRAS PREDEFINIDAS
    # --------------------------
    def count_predefined_frequencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Une todos los abstracts y cuenta apariciones (token simple) de las frases predefinidas.
        Devuelve DataFrame con columnas: ['Palabra', 'Frecuencia'] ordenado desc.
        """
        texts = df['abstract'].dropna().astype(str).apply(self._clean_text)
        concatenated = " ".join(texts.tolist())
        word_counts = Counter(concatenated.split())

        rows = []
        for phrase in self.predefined_words:
            tokens = [w for w in re.split(r'\s+', phrase.lower()) if w]
            freq = sum(word_counts.get(tok, 0) for tok in tokens)
            rows.append({"Palabra": phrase, "Frecuencia": int(freq)})

        result_df = pd.DataFrame(rows).sort_values("Frecuencia", ascending=False).reset_index(drop=True)
        return result_df

    # --------------------------
    # DESCUBRIMIENTO CON TF-IDF
    # --------------------------
    def discover_new_terms(self, df: pd.DataFrame, top_n: int = 15, max_features: int = 500) -> pd.DataFrame:
        """
        Extrae términos más relevantes según TF-IDF a partir de los abstracts.
        Devuelve DataFrame con ['Palabra', 'Score_TFIDF'].
        """
        texts = df['abstract'].dropna().astype(str).apply(self._clean_text).tolist()
        if not texts:
            return pd.DataFrame(columns=["Palabra", "Score_TFIDF"])

        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1,2))
        try:
            X = vectorizer.fit_transform(texts)
        except ValueError:
            # Corpus muy pequeño o vacío
            return pd.DataFrame(columns=["Palabra", "Score_TFIDF"])

        scores = np.asarray(X.sum(axis=0)).flatten()
        terms = vectorizer.get_feature_names_out()

        if len(terms) == 0:
            return pd.DataFrame(columns=["Palabra", "Score_TFIDF"])

        top_indices = np.argsort(scores)[::-1][:top_n]
        new_terms = [(terms[i], float(scores[i])) for i in top_indices if scores[i] > 0]

        df_new = pd.DataFrame(new_terms, columns=["Palabra", "Score_TFIDF"])
        return df_new.reset_index(drop=True)

    # --------------------------
    # EVALUAR PRECISIÓN SEMÁNTICA (USANDO SBERT SI ESTÁ DISPONIBLE)
    # --------------------------
    def evaluate_precision(self, new_terms_df: pd.DataFrame) -> pd.DataFrame:
        """
        Si SBERT está disponible: calcula similitud máxima entre cada nueva palabra
        y el set de palabras predefinidas. Añade columnas:
          - 'Similitud_max' (0..1)
          - 'Precision_%' (0..100)
        Si SBERT no está disponible, llena con NaNs y retorna.
        """
        if new_terms_df is None or len(new_terms_df) == 0:
            return new_terms_df

        # Cargar SBERT si no está cargado
        self._ensure_sbert()

        if self._sbert_model is None or self._sbert_util is None:
            # No hay modelo -> devolver NaNs con aviso
            new_terms_df = new_terms_df.copy()
            new_terms_df["Similitud_max"] = np.nan
            new_terms_df["Precisión (%)"] = np.nan
            logger.warning("SBERT no disponible: no se calcularon similitudes semánticas.")
            return new_terms_df

        # Encode (usamos batch encode)
        predefined_emb = self._sbert_model.encode(self.predefined_words, convert_to_tensor=True)
        new_emb = self._sbert_model.encode(new_terms_df["Palabra"].tolist(), convert_to_tensor=True)

        # matriz de similitud
        sim_matrix = self._sbert_util.cos_sim(new_emb, predefined_emb).cpu().numpy()
        max_sims = sim_matrix.max(axis=1)  # similitud máxima con alguna predefinida

        new_terms_df = new_terms_df.copy()
        new_terms_df["Similitud_max"] = max_sims
        new_terms_df["Precisión (%)"] = (max_sims * 100).round(2)
        return new_terms_df.sort_values("Similitud_max", ascending=False).reset_index(drop=True)

    # --------------------------
    # FLUJO COMPLETO
    # --------------------------
    def analyze_frequencies(self, df: pd.DataFrame, output_dir: str = "data/outputs", top_n: int = 15):
        """
        Ejecuta el flujo completo:
          1) contar predefinidas
          2) descubrir nuevas (TF-IDF)
          3) evaluar precisión semántica (SBERT si está)
        Guarda CSVs en output_dir y retorna (predefined_df, precision_df)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        predefined_df = self.count_predefined_frequencies(df)
        new_terms_df = self.discover_new_terms(df, top_n=top_n)
        precision_df = self.evaluate_precision(new_terms_df)

        # Guardar archivos (evitar errores de I/O con try/except)
        try:
            predefined_df.to_csv(Path(output_dir) / "frequency_words.csv", index=False)
            precision_df.to_csv(Path(output_dir) / "new_words_precision.csv", index=False)
            logger.info(f"Resultados guardados en {output_dir}")
        except Exception as e:
            logger.warning(f"No se pudieron guardar los archivos CSV: {e}")

        return predefined_df, precision_df
