import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
from motores.motor_base import MotorBase
from configuraciones.experimentos import ConfigExperimento
from typing import Any

# Función auxiliar para congelar el reloj hasta que la GPU termine
def sincronizar_hardware(hardware: str):
    if hardware == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hardware == 'mps' and torch.backends.mps.is_available():
        torch.mps.synchronize()
    # Si es 'cpu', no hacemos nada porque la CPU ya es síncrona por naturaleza

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
        self.modelo = AutoModelForCausalLM.from_pretrained(self.config.nombre_modelo,torch_dtype="auto", device_map=self.config.hardware)
        self.tokenizer.padding_side = "left" #evitamos un warning que sale en algunos modelos
        self.modelo.eval()
        return self.modelo

    def generar_respuesta(self, prompts: list[list[dict[str, str]]], max_tokens: int) -> list[dict[str, Any]]:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer.apply_chat_template(
            prompts, 
            return_tensors="pt", 
            padding=(len(prompts)>1), 
            return_dict=True, 
            add_generation_prompt=True # Vital para que el asistente empiece a hablar
        ).to(self.config.hardware)

        medidor_ttft = MedidorTTFT()
        procesadores = LogitsProcessorList([medidor_ttft])
        
        sincronizar_hardware(self.config.hardware)

        tiempo_inicio = time.time()

        with torch.inference_mode():
            outputs = self.modelo.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                temperature=0.0,
                do_sample=False, 
                logits_processor=procesadores # Le inyectamos el espía a la generación
            )
            
        sincronizar_hardware(self.config.hardware)

        # Calculamos cuánto tardó en salir el primer token
        if medidor_ttft.tiempo_primer_token is not None:
            ttft_lote = medidor_ttft.tiempo_primer_token - tiempo_inicio
        else:
            ttft_lote = time.time() - tiempo_inicio # Fallback si no generó nada
            
        resultados = []

        tokens_prefill = inputs['input_ids'].shape[1]
        
        for i in range(len(prompts)):
            tokens_solo_respuesta = outputs[i][tokens_prefill:]
            codigo_generado = self.tokenizer.decode(tokens_solo_respuesta, skip_special_tokens=True)
            print(f"Respuesta: {codigo_generado}")
            resultados.append({
                "texto": codigo_generado,
                "tokens_prompt": tokens_prefill,
                "tokens_generados": len(tokens_solo_respuesta),
                "ttft": ttft_lote
            })
            
        return resultados
