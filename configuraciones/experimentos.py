from dataclasses import dataclass
from typing import Optional

@dataclass
class ConfigExperimento:
    """
    Ficha técnica inmutable de lo que vamos a ejecutar.
    """
    nombre_modelo: str
    archivo_gguf: Optional[str] = None
    hardware: str = "cpu"             # "cuda", "mps", "cpu"
    nombre_hardware: Optional[str] = None
    motor: str = "hf"                 # "hf", "vllm", "llamacpp"
    tarea: str = "humaneval"          # "humaneval", "mmlu", etc.
    max_tokens: int = 256
    batch_size: int = 16
    cuantizacion: bool = False 
