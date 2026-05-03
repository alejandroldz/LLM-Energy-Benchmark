from abc import ABC, abstractmethod
from typing import  Any
from configuraciones.experimentos import ConfigExperimento

class MotorBase(ABC):
    """
    Molde para cualquier motor de inferencia (HuggingFace, vLLM, llama.cpp).
    """
    
    def __init__(self, config: ConfigExperimento):
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
    def generar_respuesta(self, prompts: list[list[dict[str, str]]], max_tokens: int) -> list[dict[str, Any]]:
        """
        Recibe un texto de entrada y devuelve la predicción.
        Debe devolver un diccionario con este formato exacto:
        {
            "texto": "respuesta del modelo...",
            "tokens_prompt": 150
            "tokens_generados": 42
            "ttft": 0.45
        }
        """
        pass