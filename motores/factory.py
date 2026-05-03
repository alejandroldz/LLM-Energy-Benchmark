from configuraciones.experimentos import ConfigExperimento
def crear_motor(config: ConfigExperimento):
    """
    Fábrica que decide qué motor instanciar basándose en la configuración.
    """
    if config.motor == "hf":
        from motores.motor_hf import MotorHuggingFace
        return MotorHuggingFace(config)
    elif config.motor == "vllm":
        from motores.motor_vllm import MotorVLLM
        return MotorVLLM(config)  
    elif config.motor == "llamacpp":
        from motores.motor_llamacpp import MotorLlamaCPP
        return MotorLlamaCPP(config)        
    else:
        raise ValueError(f"Motor desconocido: {config.motor}")
