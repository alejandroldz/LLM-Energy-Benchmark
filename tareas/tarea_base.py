from abc import ABC, abstractmethod


class TareaBase(ABC):
    """
    Base para cualquier benchmark.
    """

    @abstractmethod
    def cargar_datos(self) -> list:
        """
        Lee el dataset.
        Devuelve una lista con los problemas a resolver.
        """
        pass

    @abstractmethod
    def construir_prompt(self, item):
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