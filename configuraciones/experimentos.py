from dataclasses import dataclass
from typing import Optional

@dataclass
class ConfigExperimento:
    """
    Ficha técnica inmutable de lo que vamos a ejecutar.
    """
    nombre_modelo: str
    archivo_gguf: Optional[str] = None
    hardware: str = "cuda"             # "cuda", "mps", "cpu"
    nombre_hardware: Optional[str] = None
    motor: str = "vllm"                 # "hf", "vllm", "llamacpp"
    tarea: str = "humaneval"          # "humaneval", "mmlu", etc.
    max_tokens: int = 256
    batch_size: int = 16
    cuantizacion: Optional[str] = None  # None, "fp8", "int8", "fp4", "nf4", "nf4_double"
    attention_implementation: Optional[str] = None  # "triton", "flash_attention", etc. 
    speculative_decoding: dict = None  # {"method": "draft_model", "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct"} o {"method": "ngram", "model": None}
