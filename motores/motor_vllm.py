from vllm import LLM, SamplingParams
from motores.motor_base import MotorBase
from configuraciones.experimentos import ConfigExperimento
from typing import Any

class MotorVLLM(MotorBase):
    """
    Motor de inferencia vLLM
    """
    
    def cargar_modelo(self):
        self.modelo = LLM(
            model=self.config.nombre_modelo, 
            max_model_len= 1024,
            gpu_memory_utilization= 0.75,
            attention_config = {"backend": "TRITON_ATTN"} #mi ordenador tiene una gpu antigua y con configuraciones de atencion mas modernas se bloquea, con esta configuracion se soluciona el problema
        )
        return self.modelo

    def generar_respuesta(self, prompts: list[list[dict[str, str]]], max_tokens: int) -> list[dict[str, Any]]:
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.modelo.chat(prompts, params)
        resultados = []

        for output in outputs:
            texto_completo = output.outputs[0].text
            tokens_generados = len(output.outputs[0].token_ids)
            tokens_prompt = len(output.prompt_token_ids)
            ttft = 0.0
            if hasattr(output.metrics, 'first_token_time') and output.metrics.first_token_time is not None:
                ttft = float(output.metrics.first_token_time - output.metrics.first_scheduled_time)
            resultados.append({
                "texto": texto_completo,
                "tokens_prompt": tokens_prompt,
                "tokens_generados": tokens_generados,
                "ttft": ttft
            })

        return resultados   
