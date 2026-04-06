from tareas.tarea_base import TareaBase
from datasets import load_dataset
import re
import random
class TareaMMLU(TareaBase):
    """
    Evaluador para el benchmark MMLU (Massive Multitask Language Understanding).
    Adaptado a nuestra arquitectura modular para mantener el control de los tokens.
    """
    def __init__(self):
        # Aquí guardaremos las respuestas correctas para corregir el examen luego
        self.datos_reales = []

    def cargar_datos(self, max_por_grupo: int = 50):
        print("Descargando dataset MMLU...")
        dataset = load_dataset("cais/mmlu", "all", split="test")
        seed = 0
        random.seed(seed)

        # Agrupamos por subject
        grupos = {}

        for item in dataset:
            subject = item["subject"]
            grupos.setdefault(subject, []).append(item)

        self.datos_reales = []
        contador_global = 0

        for subject, items in grupos.items():
            # Muestreamos
            if len(items) > max_por_grupo:
                seleccion = random.sample(items, max_por_grupo)
            else:
                seleccion = items

            for item in seleccion:
                item_dict = dict(item)
                item_dict["task_id"] = f"MMLU/{contador_global}"
                self.datos_reales.append(item_dict)
                contador_global += 1

        print(f"Total ejemplos tras muestreo: {len(self.datos_reales)}")
        return self.datos_reales

    def construir_prompt(self, item):
        # MMLU es un examen tipo test. Construimos la pregunta.
        pregunta = item['question']
        opciones = item['choices']
        letras = ['A', 'B', 'C', 'D']
        
        # Construimos el bloque de la pregunta para el usuario
        prompt = f"{pregunta}\n\n"
        for letra, opcion in zip(letras, opciones):
            prompt += f"{letra}. {opcion}\n"
            prompt = prompt
        instruccion_sistema = (
            "You are a strict multiple-choice test taker. "
            "Respond ONLY with the single letter of the correct answer (A, B, C, or D). "
            "No explanations."
        )

        prompt_final = [
            {"role": "system", "content": instruccion_sistema}, 
            {"role": "user", "content":"Answer the following question. Write just the letter of the correct answer: " + prompt + "Answer:"}
        ]
        
        return prompt_final

    def evaluar(self, predicciones, nombre_modelo):
        aciertos = 0
        total = len(self.datos_reales)
        opciones = ['A', 'B', 'C', 'D']
        for pred, real in zip(predicciones, self.datos_reales):
            # Limpiamos espacios y pasamos a mayúsculas lo que respondió la IA
            respuesta_ia = pred["completion"].strip().upper()
            
            # La respuesta correcta viene como un número (0=A, 1=B, 2=C, 3=D)
            indice_correcto = real['answer']
            letra_correcta = opciones[indice_correcto]

            # Comprobamos 
            match = re.search(r'\b([A-D])\b', respuesta_ia)
            
            if match:
                letra_extraida = match.group(1)
                if letra_extraida == letra_correcta:
                    aciertos += 1

        nota = (aciertos / total) * 100
        print(f"Precisión del modelo '{nombre_modelo}' en MMLU: {nota:.2f}%")
        return nota