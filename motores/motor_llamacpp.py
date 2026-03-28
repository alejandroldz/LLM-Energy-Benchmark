from motores.motor_base import MotorBase
from llama_cpp import Llama
from typing import Dict, Any

class MotorLlamaCPP(MotorBase):
    """
    Motor para archivos GGUF. 
    Espera en config.nombre_modelo el formato: 'Autor/Repo/Archivo.gguf'
    """
    
    def cargar_modelo(self):
        # Ejemplo: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.q4_k_m.gguf"
        string_completo = self.config.nombre_modelo
        try:
            # Separamos por la barra: las primeras dos partes son el repo, la última el archivo
            partes = string_completo.split("/")
            repo_id = f"{partes[0]}/{partes[1]}"
            filename = partes[2]
        except IndexError:
            raise ValueError(
                "El formato de nombre_modelo para llama.cpp debe ser: 'Autor/Repositorio/archivo.gguf'\n"
                f"Recibido: {string_completo}"
            )
        acelerador = self.config.hardware
        if acelerador in ['cuda', 'mps']:
        # Activamos la GPU
            gpu_layers = -1 
        elif acelerador == 'cpu':
            # Forzamos CPU
            gpu_layers = 0
        else:
            raise ValueError(f"Acelerador {acelerador} no soportado.")
        
        modelo = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=2048, #ventana de contexto
            n_gpu_layers=gpu_layers, #gpu o cpu
            seed=42,
            verbose=False
        )
        return modelo

    def generar_respuesta(self, prompts: list[str], max_tokens: int) -> list[Dict[str, Any]]:
        resultados = []
        for prompt in prompts:
            output = self.modelo(
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                stop=[self.modelo.detokenize([self.modelo.token_eos()]).decode('utf-8')],
                echo=False
            )
            resultados.append({
                "texto": output['choices'][0]['text'],
                "tokens_generados": output['usage']['completion_tokens']
            })
        return resultados