from abc import ABC, abstractmethod
from typing import Dict, Any

class MotorBase(ABC):
    """
    Molde para cualquier motor de inferencia (HuggingFace, vLLM, llama.cpp).
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Guardamos la configuración (hardware, cuantización, ruta del modelo)
        self.config = config
        # Obligamos a que el modelo se cargue en memoria nada más instanciar la clase
        self.modelo = self.cargar_modelo()

    @abstractmethod
    def cargar_modelo(self):
        """
        Lógica interna para cargar el modelo en RAM/VRAM.
        Debe devolver el objeto del modelo listo para usar.
        """
        pass

    @abstractmethod
    def generar_respuesta(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """
        Recibe un texto de entrada y devuelve la predicción.
        OBLIGATORIO: Debe devolver un diccionario con este formato exacto:
        {
            "texto": "respuesta del modelo...",
            "tokens_generados": 42
        }
        """
        pass