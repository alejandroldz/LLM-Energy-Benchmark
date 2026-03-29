from vllm import LLM, SamplingParams
from motores.motor_base import MotorBase
from configuraciones.experimentos import ConfigExperimento
from typing import Dict, Any

class MotorVLLM(MotorBase):
    """
    Motor de inferencia vLLM
    """
    
    def cargar_modelo(self):
        self.modelo = LLM(model=self.config.nombre_modelo)
        return self.modelo

    def generar_respuesta(self, prompts: list[str], max_tokens: int) -> list[Dict[str, Any]]:
        params = SamplingParams(temperature=0.0, max_new_tokens=max_tokens)
        outputs = self.modelo.generate(prompts, params)
        resultados = []

        for output in outputs:
            texto_completo = output.outputs[0].text
            tokens_generados = len(output.outputs[0].token_ids)
            tokens_prompt = len(output.prompt_token_ids)
            ttft = output.metrics.first_token_latency - output.metrics.arrival_time
            resultados.append({
                "texto": texto_completo,
                "tokens_prompt": tokens_prompt,
                "tokens_generados": tokens_generados,
                "ttft": ttft
            })

        return resultados   

