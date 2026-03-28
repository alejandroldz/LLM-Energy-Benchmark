import torch
from vllm import LLM, SamplingParams
from motores.motor_base import MotorBase
from configuraciones.experimentos import ConfigExperimento

class MotorVLLM(MotorBase):
    """
    Motor de inferencia vLLM
    """
    
    def cargar_modelo(self):
        self.modelo = LLM(model=self.config.nombre_modelo)
        return self.modelo

    def generar_respuesta(self, prompts: list[str], max_tokens: int) -> dict:
        params = SamplingParams(temperature=0.0, max_new_tokens=max_tokens)
        outputs = self.modelo.generate(prompts, params)
        resultados = []

        for output in outputs:
            texto_completo = output.outputs[0].text
            tokens_generados = len(output.outputs[0].tokens_ids)
            resultados.append({
                "texto": texto_completo,
                "tokens_generados": tokens_generados
            })

        return resultados   
