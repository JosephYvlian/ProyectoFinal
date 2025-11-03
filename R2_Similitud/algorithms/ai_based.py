"""
Requerimiento 2: Algoritmos de IA para Similitud Textual
Universidad del Quindío - Análisis de Algoritmos

Implementa 2 algoritmos con modelos de IA:
1. BERT Embeddings (Transformers)
2. Sentence-BERT (Optimizado para similitud)

Cada algoritmo incluye:
- Explicación técnica del modelo
- Arquitectura y funcionamiento
- Ventajas y limitaciones
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================
# ALGORITMO 5: BERT EMBEDDINGS
# ============================================================

class BERTSimilarity:
    """
    Clase para cálculo de similitud con BERT.
    Mantiene el modelo cargado para eficiencia.
    """
    
    def __init__(self):
        """Inicializa el modelo BERT."""
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """Carga el modelo BERT si no está cargado."""
        if self.model is None:
            try:
                from transformers import BertTokenizer, BertModel
                import torch
                
                logger.info("Cargando modelo BERT...")
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.model = BertModel.from_pretrained('bert-base-uncased')
                self.model.eval()  # Modo evaluación
                logger.info("Modelo BERT cargado")
                
            except ImportError:
                raise ImportError(
                    "Transformers no está instalado. "
                    "Instalar con: pip install transformers torch"
                )
    
    def calculate_similarity(self, text1: str, text2: str) -> Tuple[float, dict]:
        """
        Algoritmo 5: Similitud con BERT Embeddings
        
        EXPLICACIÓN TÉCNICA:
        ====================
        BERT (Bidirectional Encoder Representations from Transformers)
        es un modelo de lenguaje pre-entrenado que genera representaciones
        vectoriales (embeddings) contextuales de textos.
        
        ARQUITECTURA DE BERT:
        
        1. INPUT LAYER:
           - Tokenización: Texto → subpalabras (WordPiece)
           - Token embeddings (vocabulario de ~30,000 tokens)
           - Segment embeddings (distinguir oraciones)
           - Position embeddings (orden de palabras)
        
        2. TRANSFORMER ENCODER (12 capas en bert-base):
           
           Cada capa tiene:
           
           a) Multi-Head Self-Attention:
              
              Attention(Q,K,V) = softmax(QK^T / √d_k) × V
              
              Donde:
              - Q (Query), K (Key), V (Value) son proyecciones lineales
              - d_k = dimensión de las keys (64)
              - Permite que cada palabra "atienda" a todas las demás
           
           b) Feed-Forward Network:
              
              FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
              
              Red neuronal de 2 capas con activación ReLU
           
           c) Layer Normalization y Residual Connections
        
        3. OUTPUT:
           - Embedding de 768 dimensiones por token
           - [CLS] token: representación de toda la secuencia
        
        MECANISMO DE ATENCIÓN (detalle):
        
        Para cada palabra, el mecanismo calcula:
        - Qué tan relevante es cada otra palabra (scores de atención)
        - Pondera las representaciones según relevancia
        - Permite capturar dependencias a largo alcance
        
        Ejemplo visual:
        "The cat sat on the mat"
        - "cat" atiende fuertemente a "sat" (sujeto-verbo)
        - "sat" atiende a "on" (verbo-preposición)
        - "on" atiende a "mat" (preposición-objeto)
        
        PRE-ENTRENAMIENTO:
        BERT se entrena en dos tareas:
        
        1. Masked Language Model (MLM):
           - Enmascara 15% de tokens aleatoriamente
           - Predice tokens enmascarados usando contexto bidireccional
           Ejemplo: "The [MASK] sat on the mat" → predice "cat"
        
        2. Next Sentence Prediction (NSP):
           - Predice si oración B sigue a oración A
           - Aprende relaciones entre oraciones
        
        CÁLCULO DE SIMILITUD:
        
        1. Generar embeddings:
           e₁ = BERT(text1)  # Vector de 768 dimensiones
           e₂ = BERT(text2)
        
        2. Similitud del coseno:
           
           sim = (e₁ · e₂) / (||e₁|| × ||e₂||)
           
           sim = Σᵢ(e₁ᵢ × e₂ᵢ) / (√Σᵢe₁ᵢ² × √Σᵢe₂ᵢ²)
        
        VENTAJAS:
        ✓ Captura significado semántico profundo
        ✓ Maneja sinónimos y paráfrasis
        ✓ Contexto bidireccional (izquierda y derecha)
        ✓ Pre-entrenado en textos masivos (Wikipedia, BookCorpus)
        
        LIMITACIONES:
        ✗ Computacionalmente costoso
        ✗ Máximo 512 tokens de entrada
        ✗ No optimizado específicamente para similitud
        
        COMPLEJIDAD:
        - Temporal: O(n²) por capa debido a la atención
        - Espacial: O(n × d) donde d=768
        - Total: ~110M parámetros en bert-base
        
        Args:
            text1, text2: Textos a comparar
        
        Returns:
            Tuple[similitud, detalles]
        """
        
        import torch
        
        # Cargar modelo si es necesario
        self._load_model()
        
        # Paso 1: Tokenizar
        inputs1 = self.tokenizer(
            text1, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        )
        inputs2 = self.tokenizer(
            text2, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # Paso 2: Generar embeddings
        with torch.no_grad():
            outputs1 = self.model(**inputs1)
            outputs2 = self.model(**inputs2)
        
        # Paso 3: Extraer representación
        # Usamos el promedio de todos los tokens (mean pooling)
        # Alternativa: usar [CLS] token (outputs.last_hidden_state[:, 0, :])
        embedding1 = outputs1.last_hidden_state.mean(dim=1).squeeze().numpy()
        embedding2 = outputs2.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Paso 4: Calcular similitud del coseno
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        similarity = dot_product / (norm1 * norm2)
        
        # Detalles del proceso
        details = {
            'model': 'bert-base-uncased',
            'parameters': '110M',
            'embedding_dimension': 768,
            'tokens_text1': inputs1['input_ids'].shape[1],
            'tokens_text2': inputs2['input_ids'].shape[1],
            'pooling_strategy': 'mean',
            'attention_heads': 12,
            'transformer_layers': 12,
            'complexity': 'O(n² × layers)',
            'explanation': (
                f"BERT generó embeddings de 768 dims. "
                f"Text1: {inputs1['input_ids'].shape[1]} tokens, "
                f"text2: {inputs2['input_ids'].shape[1]} tokens. "
                f"Similitud coseno: {similarity:.3f}"
            )
        }
        
        return float(similarity), details


# Instancia global para reutilizar modelo
_bert_similarity_instance = None

def bert_similarity(text1: str, text2: str) -> Tuple[float, dict]:
    """
    Función auxiliar para similitud con BERT.
    Reutiliza instancia del modelo para eficiencia.
    """
    global _bert_similarity_instance
    
    if _bert_similarity_instance is None:
        _bert_similarity_instance = BERTSimilarity()
    
    return _bert_similarity_instance.calculate_similarity(text1, text2)


# ============================================================
# ALGORITMO 6: SENTENCE-BERT
# ============================================================

class SentenceBERTSimilarity:
    """
    Clase para cálculo de similitud con Sentence-BERT.
    """
    
    def __init__(self):
        """Inicializa el modelo Sentence-BERT."""
        self.model = None
    
    def _load_model(self):
        """Carga el modelo Sentence-BERT."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                logger.info("Cargando modelo Sentence-BERT...")
                # Usar modelo pequeño y rápido
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Modelo Sentence-BERT cargado")
                
            except ImportError:
                raise ImportError(
                    "Sentence-transformers no está instalado. "
                    "Instalar con: pip install sentence-transformers"
                )
    
    def calculate_similarity(self, text1: str, text2: str) -> Tuple[float, dict]:
        """
        Algoritmo 6: Similitud con Sentence-BERT
        
        EXPLICACIÓN TÉCNICA:
        ====================
        Sentence-BERT (SBERT) es una modificación de BERT optimizada
        específicamente para generar embeddings de oraciones que son
        comparables mediante similitud del coseno.
        
        PROBLEMA CON BERT ESTÁNDAR:
        - BERT no está diseñado para similitud de oraciones
        - Requiere concatenar oraciones y procesar juntas
        - Computacionalmente prohibitivo para búsqueda a gran escala
        - Ejemplo: Comparar 10,000 oraciones requiere 50M inferencias
        
        SOLUCIÓN DE SENTENCE-BERT:
        Arquitectura Siamese/Triplet Network
        
        1. ARQUITECTURA SIAMESE:
        
           ┌─────────┐
           │ Texto A │──→ BERT ──→ Pooling ──→ Embedding A ──┐
           └─────────┘                                        │
                                                            Similitud
           ┌─────────┐                                        │
           │ Texto B │──→ BERT ──→ Pooling ──→ Embedding B ──┘
           └─────────┘
           
           (Mismo BERT compartido)
        
        2. POOLING STRATEGIES:
           
           a) Mean Pooling (usado en all-MiniLM):
              e = (1/n) Σᵢ hᵢ
              Promedia todos los token embeddings
           
           b) CLS Pooling:
              e = h[CLS]
              Usa solo el token [CLS]
           
           c) Max Pooling:
              e = max(h₁, h₂, ..., hₙ)
              Máximo elemento-wise
        
        3. FUNCIÓN OBJETIVO (Training):
           
           Triplet Loss:
           L = max(0, ε - cos(u_anchor, u_positive) + cos(u_anchor, u_negative))
           
           Donde:
           - u_anchor: oración ancla
           - u_positive: oración similar (label positivo)
           - u_negative: oración diferente (label negativo)
           - ε: margen (típicamente 0.5)
           
           Objetivo: Hacer que oraciones similares estén cerca en
           el espacio de embeddings, y diferentes estén lejos.
        
        4. DATASETS DE ENTRENAMIENTO:
           - NLI (Natural Language Inference): SNLI, MultiNLI
           - STS (Semantic Textual Similarity): STS Benchmark
           - Millones de pares de oraciones etiquetadas
        
        MODELO ESPECÍFICO: all-MiniLM-L6-v2
        
        Características:
        - Basado en MiniLM (modelo destilado más pequeño)
        - 6 capas (vs 12 de BERT-base)
        - 384 dimensiones (vs 768 de BERT)
        - ~23M parámetros (vs 110M de BERT)
        - Fine-tuned en 1B pares de oraciones
        
        VENTAJAS SOBRE BERT:
        ✓ 5x más rápido
        ✓ Optimizado para similitud semántica
        ✓ Embeddings comparables directamente
        ✓ Menor uso de memoria
        ✓ Mejor rendimiento en tareas de similitud
        
        VENTAJAS SOBRE MÉTODOS CLÁSICOS:
        ✓ Captura sinónimos y paráfrasis
        ✓ Comprende negación y matices
        ✓ Invariante al orden (hasta cierto punto)
        ✓ Multilingüe (modelos disponibles)
        
        BENCHMARK (STS Tasks):
        - Métodos clásicos (TF-IDF): ~0.60 correlación
        - BERT (promedio): ~0.77 correlación
        - Sentence-BERT: ~0.85 correlación
        - Humanos: ~0.90 correlación
        
        EJEMPLO DE USO:
        
        text1 = "Un perro juega en el parque"
        text2 = "Un can se divierte en el jardín"
        
        Métodos clásicos: Baja similitud (palabras diferentes)
        Sentence-BERT: Alta similitud (significado similar)
        
        COMPLEJIDAD:
        - Temporal: O(n) por texto (procesa independientemente)
        - Espacial: O(d) donde d=384
        - Comparación: O(1) (producto punto)
        
        Args:
            text1, text2: Textos a comparar
        
        Returns:
            Tuple[similitud, detalles]
        """
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Cargar modelo si es necesario
        self._load_model()
        
        # Paso 1: Generar embeddings
        # Sentence-BERT genera embeddings directamente
        embeddings = self.model.encode([text1, text2])
        
        # Paso 2: Calcular similitud del coseno
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Detalles del proceso
        details = {
            'model': 'all-MiniLM-L6-v2',
            'parameters': '23M',
            'embedding_dimension': 384,
            'architecture': 'MiniLM (6 layers)',
            'pooling': 'mean',
            'training_data': '1B sentence pairs',
            'optimization': 'Contrastive learning',
            'speedup_vs_bert': '5x faster',
            'complexity': 'O(n)',
            'explanation': (
                f"Sentence-BERT generó embeddings de 384 dims. "
                f"Modelo optimizado para similitud semántica. "
                f"Similitud coseno: {similarity:.3f}"
            )
        }
        
        return float(similarity), details


# Instancia global
_sbert_similarity_instance = None

def sentence_bert_similarity(text1: str, text2: str) -> Tuple[float, dict]:
    """
    Función auxiliar para similitud con Sentence-BERT.
    Reutiliza instancia del modelo.
    """
    global _sbert_similarity_instance
    
    if _sbert_similarity_instance is None:
        _sbert_similarity_instance = SentenceBERTSimilarity()
    
    return _sbert_similarity_instance.calculate_similarity(text1, text2)