from vllm import LLM, SamplingParams
from motores.motor_base import MotorBase
from configuraciones.experimentos import ConfigExperimento
from typing import Any

class MotorVLLM(MotorBase):
    """
    Motor de inferencia vLLM
    """
    
    def cargar_modelo(self):
        bits = self.config.cuantizacion

        kwargs_modelo: dict[str, Any] = {
            "model": self.config.nombre_modelo
        }

        if bits in (8, 4):
            kwargs_modelo["quantization"] = "bitsandbytes"
            kwargs_modelo["dtype"] = "half"

            if bits == 8:
                kwargs_modelo["enforce_eager"] = True
                kwargs_modelo["hf_overrides"] = {
                    "quantization_config": {
                        "quant_method": "bitsandbytes",
                        "load_in_8bit": True,
                        "load_in_4bit": False
                    }
                }
            else:
                kwargs_modelo["hf_overrides"] = {
                    "quantization_config": {
                        "quant_method": "bitsandbytes",
                        "load_in_8bit": False,
                        "load_in_4bit": True
                    }
                }

        self.modelo = LLM(**kwargs_modelo)
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
            if output.metrics is not None and output.metrics.first_token_latency is not None:
                ttft = float(output.metrics.first_token_latency)
            resultados.append({
                "texto": texto_completo,
                "tokens_prompt": tokens_prompt,
                "tokens_generados": tokens_generados,
                "ttft": ttft
            })

        return resultados   
