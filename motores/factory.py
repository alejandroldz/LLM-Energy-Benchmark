from configuraciones.experimentos import ConfigExperimento
from motores.motor_hf import MotorHuggingFace
# from motores.motor_vllm import MotorVLLM 

def crear_motor(config: ConfigExperimento):
    """
    Fábrica que decide qué motor instanciar basándose en la configuración.
    """
    if config.motor == "hf":
        return MotorHuggingFace(config)
    
    elif config.motor == "vllm":
        # return MotorVLLM(config)
        raise NotImplementedError("El motor vLLM todavía no está programado.")
        
    elif config.motor == "llamacpp":
        raise NotImplementedError("El motor llama.cpp todavía no está programado.")
        
    else:
        raise ValueError(f"Motor desconocido: {config.motor}")