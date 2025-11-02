"""
Requerimiento 1: Automatización de Descarga y Unificación de Datos

Módulo principal para:
- Descarga automatizada de ACM y SAGE
- Unificación de datos
- Detección de duplicados
- Generación de archivos de salida
    
"""

from .unifier import DataUnifier

__all__ = [
    'DataUnifier',
]

__version__ = '1.0.0'
__author__ = 'Universidad del Quindío - Análisis de Algoritmos'