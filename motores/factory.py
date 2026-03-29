from configuraciones.experimentos import ConfigExperimento
from motores.motor_hf import MotorHuggingFace
from motores.motor_vllm import MotorVLLM
from motores.motor_llamacpp import MotorLlamaCPP
def crear_motor(config: ConfigExperimento):
    """
    Fábrica que decide qué motor instanciar basándose en la configuración.
    """
    if config.motor == "hf":
        return MotorHuggingFace(config)
    
    elif config.motor == "vllm":
        return MotorVLLM(config)
        
        
    elif config.motor == "llamacpp":
        return MotorLlamaCPP(config)        
    else:
        raise ValueError(f"Motor desconocido: {config.motor}")