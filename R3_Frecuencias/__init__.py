"""
R3_Frecuencias
Módulo para el Requerimiento 3 del Sistema de Análisis Bibliométrico.

Incluye la clase `FrequencyAnalyzer`, encargada de:
- Calcular la frecuencia de palabras predefinidas dentro de los abstracts.
- Detectar nuevas palabras clave mediante TF-IDF y análisis semántico (Sentence-BERT).
- Evaluar la precisión semántica de las nuevas palabras frente a las predefinidas.
- Exportar resultados y generar visualizaciones opcionales.
"""

from .frequency_analyzer import FrequencyAnalyzer

__all__ = ["FrequencyAnalyzer"]
