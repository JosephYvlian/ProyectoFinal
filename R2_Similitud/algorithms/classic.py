"""
Requerimiento 2: Algoritmos Clásicos de Similitud Textual
Universidad del Quindío - Análisis de Algoritmos

Implementa 4 algoritmos clásicos:
1. Distancia de Levenshtein (edición)
2. Similitud de Jaccard (conjuntos)
3. Similitud del Coseno con TF-IDF (vectorización)
4. Coeficiente de Dice (conjuntos)

Cada algoritmo incluye:
- Explicación matemática detallada
- Implementación paso a paso
- Análisis de complejidad
"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================
# ALGORITMO 1: DISTANCIA DE LEVENSHTEIN
# ============================================================

def levenshtein_similarity(text1: str, text2: str) -> Tuple[float, dict]:
    """
    Algoritmo 1: Distancia de Levenshtein (Edit Distance)
    
    EXPLICACIÓN MATEMÁTICA:
    =======================
    La distancia de Levenshtein mide el número mínimo de operaciones
    de un solo carácter necesarias para transformar una cadena en otra.
    
    OPERACIONES PERMITIDAS:
    - Inserción: agregar un carácter
    - Eliminación: quitar un carácter
    - Sustitución: reemplazar un carácter
    
    FÓRMULA RECURSIVA:
    
    lev(a,b) = |a|                              si |b| = 0
    lev(a,b) = |b|                              si |a| = 0
    lev(a,b) = lev(tail(a), tail(b))           si a[0] = b[0]
    lev(a,b) = 1 + min {
        lev(tail(a), b),          # eliminación
        lev(a, tail(b)),          # inserción
        lev(tail(a), tail(b))     # sustitución
    }
    
    IMPLEMENTACIÓN CON PROGRAMACIÓN DINÁMICA:
    
    Matriz D de tamaño (m+1) x (n+1) donde:
    - m = longitud de text1
    - n = longitud de text2
    
    D[i][j] = distancia entre text1[0:i] y text2[0:j]
    
    Caso base:
    D[0][j] = j  (insertar j caracteres)
    D[i][0] = i  (eliminar i caracteres)
    
    Recursión:
    D[i][j] = min {
        D[i-1][j] + 1,           # eliminación
        D[i][j-1] + 1,           # inserción
        D[i-1][j-1] + cost       # sustitución (cost=0 si igual, 1 si diferente)
    }
    
    COMPLEJIDAD:
    - Temporal: O(m × n)
    - Espacial: O(m × n)
    
    NORMALIZACIÓN:
    Para obtener similitud en rango [0,1]:
    similitud = 1 - (distancia / max(len(text1), len(text2)))
    
    Args:
        text1, text2: Textos a comparar
    
    Returns:
        Tuple[similitud, detalles]
        - similitud: float en [0,1] donde 1 = idénticos
        - detalles: dict con información del proceso
    """
    
    # Paso 1: Inicialización
    m, n = len(text1), len(text2)
    
    # Matriz de programación dinámica
    D = np.zeros((m + 1, n + 1), dtype=int)
    
    # Paso 2: Casos base
    for i in range(m + 1):
        D[i][0] = i  # Eliminar i caracteres
    for j in range(n + 1):
        D[0][j] = j  # Insertar j caracteres
    
    # Paso 3: Llenar matriz con programación dinámica
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Cost = 0 si caracteres son iguales, 1 si son diferentes
            cost = 0 if text1[i-1] == text2[j-1] else 1
            
            D[i][j] = min(
                D[i-1][j] + 1,      # Eliminación
                D[i][j-1] + 1,      # Inserción
                D[i-1][j-1] + cost  # Sustitución
            )
    
    # Paso 4: Distancia final
    distance = D[m][n]
    
    # Paso 5: Normalizar a similitud [0,1]
    max_len = max(m, n)
    similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
    
    # Detalles del proceso
    details = {
        'distance': int(distance),
        'length_text1': m,
        'length_text2': n,
        'max_length': max_len,
        'operations': int(distance),
        'complexity': f"O({m} × {n})",
        'explanation': (
            f"Se requieren {distance} operaciones para transformar text1 en text2. "
            f"Con longitudes {m} y {n}, la similitud normalizada es {similarity:.3f}."
        )
    }
    
    return similarity, details


# ============================================================
# ALGORITMO 2: SIMILITUD DE JACCARD
# ============================================================

def jaccard_similarity(text1: str, text2: str) -> Tuple[float, dict]:
    """
    Algoritmo 2: Similitud de Jaccard (Índice de Jaccard)
    
    EXPLICACIÓN MATEMÁTICA:
    =======================
    El índice de Jaccard mide la similitud entre dos conjuntos
    como la razón entre la intersección y la unión.
    
    FÓRMULA:
    
    J(A,B) = |A ∩ B| / |A ∪ B|
    
    Equivalentemente:
    J(A,B) = |A ∩ B| / (|A| + |B| - |A ∩ B|)
    
    Donde:
    - A, B son conjuntos de tokens (palabras)
    - A ∩ B: elementos en común (intersección)
    - A ∪ B: todos los elementos únicos (unión)
    - | | denota cardinalidad (tamaño del conjunto)
    
    PROPIEDADES:
    1. 0 ≤ J(A,B) ≤ 1
    2. J(A,A) = 1  (conjunto igual a sí mismo)
    3. J(A,B) = 0  si A ∩ B = ∅ (sin elementos comunes)
    4. Simétrico: J(A,B) = J(B,A)
    
    EJEMPLO:
    text1 = "machine learning is fascinating"
    text2 = "deep learning is interesting"
    
    A = {machine, learning, is, fascinating}
    B = {deep, learning, is, interesting}
    
    A ∩ B = {learning, is}                     → 2 elementos
    A ∪ B = {machine, learning, is, fascinating, deep, interesting} → 6 elementos
    
    J(A,B) = 2/6 = 0.333
    
    COMPLEJIDAD:
    - Temporal: O(n + m) donde n=|A|, m=|B|
    - Espacial: O(n + m)
    
    Args:
        text1, text2: Textos a comparar
    
    Returns:
        Tuple[similitud, detalles]
    """
    
    # Paso 1: Tokenización (convertir a conjuntos de palabras)
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    # Paso 2: Intersección (palabras en común)
    intersection = tokens1.intersection(tokens2)
    
    # Paso 3: Unión (todas las palabras únicas)
    union = tokens1.union(tokens2)
    
    # Paso 4: Calcular índice de Jaccard
    if len(union) == 0:
        similarity = 0.0
    else:
        similarity = len(intersection) / len(union)
    
    # Detalles del proceso
    details = {
        'tokens_text1': len(tokens1),
        'tokens_text2': len(tokens2),
        'intersection_size': len(intersection),
        'union_size': len(union),
        'intersection_tokens': sorted(list(intersection))[:10],  # Primeros 10
        'complexity': f"O({len(tokens1)} + {len(tokens2)})",
        'explanation': (
            f"De {len(tokens1)} y {len(tokens2)} tokens únicos, "
            f"{len(intersection)} son comunes. "
            f"Unión total: {len(union)} tokens. "
            f"Jaccard: {len(intersection)}/{len(union)} = {similarity:.3f}"
        )
    }
    
    return similarity, details


# ============================================================
# ALGORITMO 3: SIMILITUD DEL COSENO CON TF-IDF
# ============================================================

def cosine_tfidf_similarity(text1: str, text2: str) -> Tuple[float, dict]:
    """
    Algoritmo 3: Similitud del Coseno con TF-IDF
    
    EXPLICACIÓN MATEMÁTICA:
    =======================
    Combina dos técnicas:
    1. TF-IDF: Ponderación de términos
    2. Similitud del Coseno: Comparación vectorial
    
    PARTE 1: TF-IDF (Term Frequency - Inverse Document Frequency)
    
    TF(t,d) = frecuencia del término t en documento d
           = (# veces que t aparece en d) / (# total de palabras en d)
    
    IDF(t,D) = logaritmo de frecuencia inversa de documento
            = log(N / |{d ∈ D : t ∈ d}|)
    
    Donde:
    - N = número total de documentos
    - |{d ∈ D : t ∈ d}| = número de documentos que contienen t
    
    TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)
    
    INTUICIÓN:
    - TF: Palabras frecuentes en un documento son importantes
    - IDF: Palabras raras en la colección son más discriminativas
    - TF-IDF alto → término importante y específico
    
    PARTE 2: SIMILITUD DEL COSENO
    
    cos(θ) = (A · B) / (||A|| × ||B||)
    
    Donde:
    - A, B son vectores TF-IDF
    - A · B = producto punto = Σ(Aᵢ × Bᵢ)
    - ||A|| = norma euclidiana = √(Σ Aᵢ²)
    - θ = ángulo entre vectores
    
    FORMA EXPANDIDA:
    
    cos(θ) = Σ(Aᵢ × Bᵢ) / (√(Σ Aᵢ²) × √(Σ Bᵢ²))
    
    INTERPRETACIÓN GEOMÉTRICA:
    - cos(0°) = 1   → vectores idénticos (paralelos)
    - cos(90°) = 0  → vectores ortogonales (independientes)
    - 0 ≤ cos(θ) ≤ 1 (para vectores con componentes positivos)
    
    VENTAJAS:
    - Invariante a la longitud del documento
    - Considera importancia de palabras (IDF)
    - Ampliamente usado en recuperación de información
    
    COMPLEJIDAD:
    - Temporal: O(n) donde n = dimensión del vocabulario
    - Espacial: O(n)
    
    Args:
        text1, text2: Textos a comparar
    
    Returns:
        Tuple[similitud, detalles]
    """
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Paso 1: Crear vectorizador TF-IDF
    vectorizer = TfidfVectorizer()
    
    # Paso 2: Vectorizar ambos textos
    # Necesitamos un corpus mínimo de 2 documentos
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Paso 3: Calcular similitud del coseno
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity = similarity_matrix[0][0]
    
    # Paso 4: Extraer información del proceso
    feature_names = vectorizer.get_feature_names_out()
    tfidf_vector1 = tfidf_matrix[0].toarray()[0]
    tfidf_vector2 = tfidf_matrix[1].toarray()[0]
    
    # Términos más importantes (top 5 por documento)
    top_indices1 = tfidf_vector1.argsort()[-5:][::-1]
    top_indices2 = tfidf_vector2.argsort()[-5:][::-1]
    
    top_terms1 = [(feature_names[i], tfidf_vector1[i]) for i in top_indices1 if tfidf_vector1[i] > 0]
    top_terms2 = [(feature_names[i], tfidf_vector2[i]) for i in top_indices2 if tfidf_vector2[i] > 0]
    
    # Detalles del proceso
    details = {
        'vocabulary_size': len(feature_names),
        'vector_dimension': len(tfidf_vector1),
        'non_zero_terms_text1': np.count_nonzero(tfidf_vector1),
        'non_zero_terms_text2': np.count_nonzero(tfidf_vector2),
        'top_terms_text1': top_terms1,
        'top_terms_text2': top_terms2,
        'complexity': f"O({len(feature_names)})",
        'explanation': (
            f"Vocabulario de {len(feature_names)} términos. "
            f"Vector text1 tiene {np.count_nonzero(tfidf_vector1)} términos activos, "
            f"text2 tiene {np.count_nonzero(tfidf_vector2)}. "
            f"Similitud del coseno: {similarity:.3f}"
        )
    }
    
    return similarity, details


# ============================================================
# ALGORITMO 4: COEFICIENTE DE DICE
# ============================================================

def dice_coefficient(text1: str, text2: str) -> Tuple[float, dict]:
    """
    Algoritmo 4: Coeficiente de Dice (Sørensen-Dice)
    
    EXPLICACIÓN MATEMÁTICA:
    =======================
    El coeficiente de Dice es una métrica de similitud entre conjuntos,
    similar a Jaccard pero con diferente ponderación.
    
    FÓRMULA:
    
    Dice(A,B) = 2 × |A ∩ B| / (|A| + |B|)
    
    Donde:
    - A, B son conjuntos de tokens
    - |A ∩ B| = tamaño de la intersección
    - |A|, |B| = tamaños de los conjuntos
    
    RELACIÓN CON JACCARD:
    
    Dice(A,B) = 2 × Jaccard(A,B) / (1 + Jaccard(A,B))
    
    O inversamente:
    Jaccard(A,B) = Dice(A,B) / (2 - Dice(A,B))
    
    DIFERENCIAS CON JACCARD:
    - Dice da más peso a la intersección (factor de 2)
    - Dice siempre ≥ Jaccard
    - Dice es más "optimista" sobre la similitud
    
    EJEMPLO:
    A = {a, b, c}
    B = {b, c, d}
    
    |A ∩ B| = 2  (elementos: b, c)
    |A| = 3
    |B| = 3
    
    Dice(A,B) = 2×2 / (3+3) = 4/6 = 0.667
    Jaccard(A,B) = 2 / (3+3-2) = 2/4 = 0.500
    
    PROPIEDADES:
    1. 0 ≤ Dice(A,B) ≤ 1
    2. Dice(A,A) = 1
    3. Dice(A,B) = 0 si A ∩ B = ∅
    4. Simétrico: Dice(A,B) = Dice(B,A)
    
    APLICACIONES:
    - Bioinformática (comparación de secuencias)
    - Procesamiento de imágenes
    - Evaluación de segmentación
    
    COMPLEJIDAD:
    - Temporal: O(n + m)
    - Espacial: O(n + m)
    
    Args:
        text1, text2: Textos a comparar
    
    Returns:
        Tuple[similitud, detalles]
    """
    
    # Paso 1: Tokenización
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    # Paso 2: Intersección
    intersection = tokens1.intersection(tokens2)
    
    # Paso 3: Calcular coeficiente de Dice
    denominator = len(tokens1) + len(tokens2)
    if denominator == 0:
        similarity = 0.0
    else:
        similarity = (2 * len(intersection)) / denominator
    
    # Comparación con Jaccard (para contexto)
    union = tokens1.union(tokens2)
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
    
    # Detalles del proceso
    details = {
        'tokens_text1': len(tokens1),
        'tokens_text2': len(tokens2),
        'intersection_size': len(intersection),
        'dice_numerator': 2 * len(intersection),
        'dice_denominator': denominator,
        'jaccard_for_comparison': jaccard,
        'dice_vs_jaccard_ratio': similarity / jaccard if jaccard > 0 else 0,
        'complexity': f"O({len(tokens1)} + {len(tokens2)})",
        'explanation': (
            f"Intersección de {len(intersection)} tokens. "
            f"Dice = 2×{len(intersection)} / ({len(tokens1)}+{len(tokens2)}) = {similarity:.3f}. "
            f"Jaccard equivalente: {jaccard:.3f}."
        )
    }
    
    return similarity, details