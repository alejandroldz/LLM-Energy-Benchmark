from dataclasses import dataclass
from typing import Optional

@dataclass
class ConfigExperimento:
    """
    Ficha técnica inmutable de lo que vamos a ejecutar.
    """
    nombre_modelo: str
    hardware: str = "cpu"             # "cuda", "mps", "cpu"
    nombre_hardware: Optional[str] = None
    motor: str = "hf"                 # "hf", "vllm", "llamacpp"
    tarea: str = "humaneval"          # "humaneval", "mmlu", etc.
    cuantizacion: Optional[str] = None  # "4bit", "8bit", None
    max_tokens: int = 256