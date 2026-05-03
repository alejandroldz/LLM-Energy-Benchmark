import gc
import os
import time
from typing import Any

from llama_cpp import Llama, LogitsProcessorList, LogitsProcessor
from motores.motor_base import MotorBase


class MedidorTTFT(LogitsProcessor):
    def __init__(self):
        self.tiempo_primer_token = None

    def __call__(self, input_ids, scores):
        if self.tiempo_primer_token is None:
            self.tiempo_primer_token = time.time()
        return scores


class MotorLlamaCPP(MotorBase):
    """
    Motor para archivos GGUF.
    Acepta tanto archivos locales como rutas de HuggingFace:
    'Autor/Repo/Archivo.gguf'
    """

    def _crear_modelo(self, string_completo: str, gpu_layers: int) -> Llama:
        if os.path.exists(string_completo):
            print(f"Cargando modelo local desde: {string_completo} con n_gpu_layers={gpu_layers}")
            return Llama(
                model_path=string_completo,
                n_ctx=4096,
                n_gpu_layers=gpu_layers,
                seed=42,
                verbose=True,
            )

        try:
            partes = string_completo.split("/")
            repo_id = f"{partes[0]}/{partes[1]}"
            filename = partes[2]
        except IndexError:
            raise ValueError(
                f"No se encontró el archivo local '{string_completo}'.\n"
                "Si intentabas descargarlo, el formato debe ser: 'Autor/Repositorio/archivo.gguf'"
            )

        print(f"Descargando modelo desde HuggingFace: {repo_id} -> {filename} con n_gpu_layers={gpu_layers}")
        return Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=4096,
            n_gpu_layers=gpu_layers,
            seed=42,
            verbose=True,
        )

    def cargar_modelo(self):
        string_completo = self.config.archivo_gguf

        if not string_completo:
            raise ValueError(
                f"Para usar Llama.cpp con el modelo '{self.config.nombre_modelo}', "
                f"debes especificar 'archivo_gguf' en ConfigExperimento."
            )

        acelerador = self.config.hardware

        if acelerador == "cpu":
            return self._crear_modelo(string_completo, gpu_layers=0)

        if acelerador not in ["cuda", "mps"]:
            raise ValueError(f"Acelerador {acelerador} no soportado.")

        # Probamos de más GPU a menos GPU (Lo que sigue sirve para dividir el trbajo entre GPU y CPU si no cabe completo en GPU)
        capas = [i for i in range(1,50)]
        capas = capas[::-1]
        capas_a_probar = [-1]+capas
        print(capas_a_probar)
        ultimo_error = None

        for gpu_layers in capas_a_probar:
            try:
                self.modelo = self._crear_modelo(string_completo, gpu_layers=gpu_layers)
                self.gpu_layers_usado = gpu_layers
                print(f"Modelo cargado correctamente con n_gpu_layers={gpu_layers}")
                return self.modelo

            except Exception as e:
                ultimo_error = e
                print(f"Falló con n_gpu_layers={gpu_layers}: {e}")
                gc.collect()

        raise RuntimeError(
            f"No se pudo cargar el modelo tras probar varias configuraciones de GPU. "
            f"Último error: {ultimo_error}"
        )

    def generar_respuesta(self, prompts: list[list[dict[str, str]]], max_tokens: int) -> list[dict[str, Any]]:
        resultados = []

        for prompt in prompts:
            espia_ttft = MedidorTTFT()
            procesadores = LogitsProcessorList([espia_ttft])

            tiempo_inicio = time.time()

            output = self.modelo.create_chat_completion(
                messages=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                logits_processor=procesadores,
            )

            ttft = espia_ttft.tiempo_primer_token - tiempo_inicio if espia_ttft.tiempo_primer_token else 0.0

            resultados.append({
                "texto": output["choices"][0]["message"]["content"],
                "tokens_prompt": output["usage"]["prompt_tokens"],
                "tokens_generados": output["usage"]["completion_tokens"],
                "ttft": ttft,
            })

        return resultados