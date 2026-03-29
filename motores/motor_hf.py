import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
from motores.motor_base import MotorBase
from configuraciones.experimentos import ConfigExperimento
from typing import Dict, Any

class MedidorTTFT(LogitsProcessor):
    def __init__(self):
        self.tiempo_primer_token = None
        
    def __call__(self, input_ids, scores):
        # Si es la primera vez que se llama, es que acaba de generar el primer token
        if self.tiempo_primer_token is None:
            self.tiempo_primer_token = time.time()
        return scores


class MotorHuggingFace(MotorBase):
    """
    Motor de inferencia usando exactamente la lógica original de HuggingFace.
    """
    
    def cargar_modelo(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.nombre_modelo)
        self.modelo = AutoModelForCausalLM.from_pretrained(self.config.nombre_modelo).to(self.config.hardware)
        return self.modelo

    def generar_respuesta(self, prompts: list[str], max_tokens: int) -> list[Dict[str, Any]]:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.config.hardware)

        medidor_ttft = MedidorTTFT()
        procesadores = LogitsProcessorList([medidor_ttft])
        tiempo_inicio = time.time()

        with torch.no_grad():
            outputs = self.modelo.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                temperature=0.0,
                logits_processor=procesadores # Le inyectamos el espía a la generación
            )
            
        # Calculamos cuánto tardó en salir el primer token
        if medidor_ttft.tiempo_primer_token is not None:
            ttft_lote = medidor_ttft.tiempo_primer_token - tiempo_inicio
        else:
            ttft_lote = time.time() - tiempo_inicio # Fallback si no generó nada
            
        resultados = []

        # Los tokens del prompt ya los calculabas tú perfectamente
        tokens_prefill = inputs['input_ids'].shape[1]
        
        for i in range(len(prompts)):
            tokens_solo_respuesta = outputs[i][tokens_prefill:]
            codigo_generado = self.tokenizer.decode(tokens_solo_respuesta, skip_special_tokens=True)
            
            resultados.append({
                "texto": codigo_generado,
                "tokens_prompt": tokens_prefill,
                "tokens_generados": len(tokens_solo_respuesta),
                "ttft": ttft_lote
            })
            
        return resultados