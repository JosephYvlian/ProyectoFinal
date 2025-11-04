"""
Requerimiento 4: Clustering Jerárquico
Universidad del Quindío - Análisis de Algoritmos

Implementa 3 métodos de clustering jerárquico:
1. Single Linkage
2. Complete Linkage
3. Ward Linkage

Cada método genera un dendrograma y calcula su coherencia (Silhouette Score).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger(__name__)


class ClusteringAnalyzer:
    """
    Analizador de Clustering Jerárquico con 3 métodos:
    Single, Complete y Ward.
    """

    def __init__(self, output_dir: str = "data/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ===============================
    # 1️⃣ PREPROCESAMIENTO DE TEXTO
    # ===============================
    def preprocess_texts(self, texts):
        """
        Limpia y normaliza los abstracts.
        """
        import re
        import string

        cleaned = []
        for t in texts:
            t = t.lower()
            t = re.sub(r"http\S+", "", t)
            t = re.sub(r"[^a-z\s]", "", t)
            t = t.translate(str.maketrans("", "", string.punctuation))
            t = re.sub(r"\s+", " ", t).strip()
            cleaned.append(t)
        return cleaned

    # ===============================
    # 2️⃣ VECTORIZACIÓN TF-IDF
    # ===============================
    def vectorize_texts(self, texts):
        """
        Convierte los textos en vectores TF-IDF.
        """
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        X = vectorizer.fit_transform(texts)
        return X, vectorizer

    # ===============================
    # 3️⃣ CLUSTERING JERÁRQUICO
    # ===============================
    def perform_clustering(self, X, titles, method: str) -> Dict:
        """
        Ejecuta el clustering jerárquico con un método dado.
        """
        logger.info(f"→ Ejecutando clustering con método: {method}")

        # Convertir a matriz densa para Ward
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        distance_matrix = cosine_distances(X_dense)

        linkage_matrix = linkage(distance_matrix, method=method)

        # Cortar el dendrograma en un número óptimo de clústeres (opcional)
        clusters = fcluster(linkage_matrix, t=3, criterion="maxclust")

        # Calcular coherencia
        score = silhouette_score(X_dense, clusters, metric="euclidean")

        # Crear dendrograma
        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(
            linkage_matrix,
            labels=[f"{i+1}" for i in range(len(titles))],
            orientation="top",
            leaf_rotation=90,
            leaf_font_size=10,
            color_threshold=0.7 * max(linkage_matrix[:, 2]),
            ax=ax
        )
        ax.set_title(f"Dendrograma ({method.capitalize()} Linkage)")
        ax.set_xlabel("Abstracts")
        ax.set_ylabel("Distancia")

        # Guardar gráfico
        file_path = self.output_dir / f"dendrogram_{method.lower()}.png"
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

        logger.info(f"✓ Dendrograma guardado: {file_path}")

        return {
            "method": method,
            "silhouette": round(score, 3),
            "n_clusters": len(set(clusters)),
            "file": str(file_path)
        }

    # ===============================
    # 4️⃣ PIPELINE COMPLETO
    # ===============================
    def analyze_clustering(self, df: pd.DataFrame, n_samples: int = 50):
        """
        Ejecuta el análisis completo con los 3 métodos.
        """
        logger.info("Iniciando análisis de clustering...")

        df = df[df["abstract"].notna() & (df["abstract"].str.len() > 50)].head(n_samples)
        texts = df["abstract"].tolist()
        titles = df["title"].tolist()

        cleaned_texts = self.preprocess_texts(texts)
        X, _ = self.vectorize_texts(cleaned_texts)

        methods = ["single", "complete", "ward"]
        results = []

        for m in methods:
            result = self.perform_clustering(X, titles, m)
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "clustering_results.csv", index=False)

        best = results_df.loc[results_df["silhouette"].idxmax()]

        print("\n" + "=" * 70)
        print("RESULTADOS DEL CLUSTERING JERÁRQUICO")
        print("=" * 70)
        print(results_df.to_string(index=False))
        print("\n✓ Mejor método:", best["method"].capitalize(),
              f"(Silhouette Score: {best['silhouette']})")

        return results_df
