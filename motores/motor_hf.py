import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from motores.motor_base import MotorBase
from configuraciones.experimentos import ConfigExperimento

class MotorHuggingFace(MotorBase):
    """
    Motor de inferencia usando exactamente la lógica original de HuggingFace.
    """
    
    def cargar_modelo(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.nombre_modelo)
        self.modelo = AutoModelForCausalLM.from_pretrained(self.config.nombre_modelo).to(self.config.hardware)
        return self.modelo

    def generar_respuesta(self, prompt: str, max_tokens: int) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.hardware)
        with torch.no_grad():
            outputs = self.modelo.generate(**inputs, max_new_tokens=max_tokens, temperature=0.0)
            
        tokens_prefill = inputs['input_ids'].shape[1]
        tokens_solo_respuesta = outputs[0][tokens_prefill:] 
        
        codigo_generado = self.tokenizer.decode(tokens_solo_respuesta, skip_special_tokens=True)   
        
        # Devolvemos el diccionario estricto que pide el MotorBase
        return {
            "texto": codigo_generado,
            "tokens_generados": len(tokens_solo_respuesta)
        }