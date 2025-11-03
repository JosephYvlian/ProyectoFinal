"""
Algoritmos de similitud textual.

Módulo que agrupa todos los algoritmos de similitud implementados
para el Requerimiento 2.

Clásicos (4):
- levenshtein_similarity: Distancia de edición
- jaccard_similarity: Similitud de conjuntos
- cosine_tfidf_similarity: Vectorización estadística con TF-IDF
- dice_coefficient: Coeficiente de Sørensen-Dice

IA (2):
- bert_similarity: Embeddings con BERT Transformers
- sentence_bert_similarity: Sentence-BERT optimizado

"""

# Importar algoritmos clásicos
from .classic import (
    levenshtein_similarity,
    jaccard_similarity,
    cosine_tfidf_similarity,
    dice_coefficient
)

# Importar algoritmos IA
from .ai_based import (
    bert_similarity,
    sentence_bert_similarity
)

# Definir qué se exporta con "from algorithms import *"
__all__ = [
    # Clásicos
    'levenshtein_similarity',
    'jaccard_similarity',
    'cosine_tfidf_similarity',
    'dice_coefficient',
    # IA
    'bert_similarity',
    'sentence_bert_similarity'
]

# Información del módulo
__version__ = '1.0.0'
__author__ = 'Universidad del Quindío - Análisis de Algoritmos'

# Diccionario con información de cada algoritmo (opcional pero útil)
ALGORITHM_INFO = {
    'levenshtein_similarity': {
        'name': 'Levenshtein',
        'type': 'Clásico',
        'category': 'Distancia de Edición',
        'complexity': 'O(m×n)',
        'description': 'Mide operaciones mínimas para transformar un texto en otro'
    },
    'jaccard_similarity': {
        'name': 'Jaccard',
        'type': 'Clásico',
        'category': 'Similitud de Conjuntos',
        'complexity': 'O(n+m)',
        'description': 'Razón entre intersección y unión de tokens'
    },
    'cosine_tfidf_similarity': {
        'name': 'Cosine + TF-IDF',
        'type': 'Clásico',
        'category': 'Vectorización Estadística',
        'complexity': 'O(n)',
        'description': 'Similitud del coseno entre vectores TF-IDF'
    },
    'dice_coefficient': {
        'name': 'Dice',
        'type': 'Clásico',
        'category': 'Similitud de Conjuntos',
        'complexity': 'O(n+m)',
        'description': 'Coeficiente de Sørensen-Dice'
    },
    'bert_similarity': {
        'name': 'BERT',
        'type': 'IA',
        'category': 'Deep Learning',
        'complexity': 'O(n²×layers)',
        'description': 'Embeddings contextuales con Transformers (110M params)'
    },
    'sentence_bert_similarity': {
        'name': 'Sentence-BERT',
        'type': 'IA',
        'category': 'Deep Learning Optimizado',
        'complexity': 'O(n)',
        'description': 'BERT optimizado para similitud (23M params)'
    }
}