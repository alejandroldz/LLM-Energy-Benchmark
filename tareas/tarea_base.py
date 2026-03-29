from abc import ABC, abstractmethod
from typing import List, Dict, Any

class TareaBase(ABC):
    """
    Molde estricto para cualquier benchmark (HumanEval, MMLU, GSM8K...).
    """

    @abstractmethod
    def cargar_datos(self) -> List[Any]:
        """
        Lee el dataset (JSONL, CSV, etc.).
        Devuelve una lista con los problemas a resolver.
        """
        pass

    @abstractmethod
    def construir_prompt(self, item: Any) -> str:
        """
        Adapta el problema al formato que necesita el LLM.
        Por ejemplo, MMLU necesita inyectar "A), B), C), D)".
        """
        pass

    @abstractmethod
    def evaluar(self, predicciones, nombre_modelo) -> float:
        """
        Compara las respuestas del LLM con las soluciones reales.
        Devuelve una nota numérica (ej: 0 a 100).
        """
        pass