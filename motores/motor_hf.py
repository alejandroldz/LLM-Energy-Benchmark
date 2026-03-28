import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from motores.motor_base import MotorBase
from configuraciones.experimentos import ConfigExperimento
from typing import Dict, Any
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

        with torch.no_grad():
            outputs = self.modelo.generate(**inputs, max_new_tokens=max_tokens, temperature=0.0)
            
        resultados = []

        tokens_prefill = inputs['input_ids'].shape[1]
        # 5. Procesamos cada respuesta generada en este lote
        for i in range(len(prompts)):
            # Aislamos solo los tokens generados por la IA
            tokens_solo_respuesta = outputs[i][tokens_prefill:]
            
            # Decodificamos a texto
            codigo_generado = self.tokenizer.decode(tokens_solo_respuesta, skip_special_tokens=True)
            # Empaquetamos la respuesta en el diccionario que pide tu arquitectura
            resultados.append({
                "texto": codigo_generado,
                "tokens_generados": len(tokens_solo_respuesta)
            })
            
        return resultados