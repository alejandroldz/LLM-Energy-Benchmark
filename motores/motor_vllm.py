from vllm import LLM, SamplingParams, RequestOutput
from motores.motor_base import MotorBase
from configuraciones.experimentos import ConfigExperimento
from typing import Any

class MotorVLLM(MotorBase):
    """
    Motor de inferencia vLLM
    """
    
    def cargar_modelo(self):
        att = self.config.attention_implementation
        if att == None:
            att = 'auto'
        
        if self.config.speculative_decoding["method"] == "draft_model":
            speculative_config = {
                "method": "draft_model",
                "model": self.config.speculative_decoding["model"],
                "num_speculative_tokens": 8,
            }
        elif self.config.speculative_decoding["method"] == "ngram":
            speculative_config = {
                "method": "ngram",
                "num_speculative_tokens": 5,
                "prompt_lookup_max": 4,
            }
        else:
            speculative_config = None
            
        quantization, extra_config = self.cargar_cuantizacion()
        
        self.modelo = LLM(
            model=self.config.nombre_modelo, 
            max_model_len= 800 if self.config.tarea != "resumen" else 2048,
            gpu_memory_utilization= 0.90,
            disable_log_stats=False,
            attention_config={"backend": att},
            speculative_config=speculative_config,
            quantization=quantization,
            # model_loader_extra_config=extra_config

        )

        return self.modelo

    def cargar_cuantizacion(self):
        if self.config.cuantizacion == "int8":
            quantization = "bitsandbytes"
            extra_config = {
                "load_in_8bit": True,
                "load_in_4bit": False,
                "llm_int8_threshold": 6.0,
            }
        elif self.config.cuantizacion == "fp4":
            quantization = "bitsandbytes"
            extra_config = {
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_quant_type": "fp4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": False,
            }
        elif self.config.cuantizacion == "nf4":
            quantization = "bitsandbytes"
            extra_config = {
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": False,
            }
        elif self.config.cuantizacion == "nf4_double":
            quantization = "bitsandbytes"
            extra_config = {
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            }

        elif self.config.cuantizacion == "fp8":
            quantization = "fp8"
            extra_config = {}
        elif self.config.cuantizacion == "gptq":
            quantization = "gptq"
            extra_config = {}

        elif self.config.cuantizacion == "awq":
            quantization = "awq"
            extra_config = {}
        else:
            quantization = None
            extra_config = {}
        
        return quantization, extra_config
        
    
    def generar_respuesta(self, prompts: list[list[dict[str, str]]], max_tokens: int) -> list[dict[str, Any]]:
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.modelo.chat(prompts, params)
        resultados = []
        print("Extrayendo resultados del lote...")
        for output in outputs:
            texto_completo = output.outputs[0].text
            tokens_generados = len(output.outputs[0].token_ids)
            tokens_prompt = len(output.prompt_token_ids)
            ttft = 0.0
            if output.metrics is not None:
                ttft = float(output.metrics.first_token_latency)
            resultados.append({
                "texto": texto_completo,
                "tokens_prompt": tokens_prompt,
                "tokens_generados": tokens_generados,
                "ttft": ttft
            })

        return resultados   
