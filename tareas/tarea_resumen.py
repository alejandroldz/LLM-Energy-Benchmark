from tareas.tarea_base import TareaBase
from datasets import load_dataset
import evaluate
import random


class TareaResumen(TareaBase):

    def __init__(self):
        self.datos_reales = []
        self.metric_rouge = evaluate.load("rouge")

    def cargar_datos(self, max_ejemplos: int = 100, seed: int = 1):
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")

        random.seed(seed)

        if max_ejemplos is not None and len(dataset) > max_ejemplos:
            indices = random.sample(range(len(dataset)), max_ejemplos)
            dataset = dataset.select(indices)

        self.datos_reales = []

        for i, item in enumerate(dataset):
            self.datos_reales.append({
                "task_id": f"RESUMEN/{i}",
                "article": item["article"],
                "highlights": item["highlights"],
            })

        return self.datos_reales

    def construir_prompt(self, item):
        articulo = item["article"]

        instruccion_sistema = (
            "You are a strict summarization engine. "
            "Write a concise summary of the given text. "
            "Do not add explanations or extra text."
        )

        prompt_final = [
            {"role": "system", "content": instruccion_sistema},
            {
                "role": "user",
                "content": (
                    "Summarize the following text in 2 to 4 sentences:\n\n"
                    f"{articulo}\n\nSummary:"
                ),
            },
        ]
        return prompt_final
    
    
    def evaluar(self, predicciones, nombre_modelo):
        pred_por_id = {
            pred["task_id"]: pred["completion"]
            for pred in predicciones
        }

        referencias = []
        generaciones = []

        for item in self.datos_reales:
            task_id = item["task_id"]
            referencias.append(item["highlights"])
            generaciones.append(pred_por_id.get(task_id, ""))

        resultados = self.metric_rouge.compute(
            predictions=generaciones,
            references=referencias,
            use_stemmer=True,
        )
        media = ((resultados["rougeL"] + resultados["rouge1"] + resultados["rouge2"])/3)*100

        return media