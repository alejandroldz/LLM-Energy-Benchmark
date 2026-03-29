import time
from motores.motor_base import MotorBase
from llama_cpp import Llama, LogitsProcessorList, LogitsProcessor
from typing import Dict, Any
import os
# Reutilizamos el mismo "espía" que en HuggingFace
class MedidorTTFT(LogitsProcessor):
    def __init__(self):
        self.tiempo_primer_token = None
        
    def __call__(self, input_ids, scores):
        # Si es nulo, significa que este es el primer token que se va a generar
        if self.tiempo_primer_token is None:
            self.tiempo_primer_token = time.time()
        return scores
class MotorLlamaCPP(MotorBase):
    """
    Motor para archivos GGUF. 
    Acepta tanto archivos locales como rutas de HuggingFace ('Autor/Repo/Archivo.gguf')
    """
    
    def cargar_modelo(self):
        string_completo = self.config.archivo_gguf

        if not string_completo:
            raise ValueError(f"Para usar Llama.cpp con el modelo '{self.config.nombre_modelo}', "
                             f"debes especificar 'archivo_gguf' en ConfigExperimento.")
            
        acelerador = self.config.hardware
        if acelerador in ['cuda', 'mps']:
            gpu_layers = -1 
        elif acelerador == 'cpu':
            gpu_layers = 0
        else:
            raise ValueError(f"Acelerador {acelerador} no soportado.")

        if os.path.exists(string_completo):
            #Si el archivo está en tu ordenador local, lo carga directo
            print(f"Cargando modelo local desde: {string_completo}")
            self.modelo = Llama(
                model_path=string_completo,
                n_ctx=2048,
                n_gpu_layers=gpu_layers,
                seed=42,
                verbose=False
            )
        else:
            # Si no es un archivo local, asume que es de HuggingFace y lo descarga
            try:
                partes = string_completo.split("/")
                repo_id = f"{partes[0]}/{partes[1]}"
                filename = partes[2]
            except IndexError:
                raise ValueError(
                    f"No se encontró el archivo local '{string_completo}'.\n"
                    "Si intentabas descargarlo, el formato debe ser: 'Autor/Repositorio/archivo.gguf'"
                )
            
            print(f"Descargando modelo desde HuggingFace: {repo_id} -> {filename}")
            self.modelo = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=2048, 
                n_gpu_layers=gpu_layers, 
                seed=42,
                verbose=False
            )
            
        return self.modelo

    def generar_respuesta(self, prompts: list[str], max_tokens: int) -> list[Dict[str, Any]]:
            resultados = []
            
            for prompt in prompts:
                espia_ttft = MedidorTTFT()
                procesadores = LogitsProcessorList([espia_ttft])
                
                tiempo_inicio = time.time()
                
                output = self.modelo(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    echo=False,
                    logits_processor=procesadores 
                )
                
                # Calculamos el TTFT
                ttft = espia_ttft.tiempo_primer_token - tiempo_inicio
                    
                tokens_prompt = output['usage']['prompt_tokens']
                tokens_generados = output['usage']['completion_tokens']
                
                resultados.append({
                    "texto": output['choices'][0]['text'],
                    "tokens_prompt": tokens_prompt,
                    "tokens_generados": tokens_generados,
                    "ttft": ttft
                })
                
            return resultados