from tareas.tarea_base import TareaBase
from datasets import load_dataset
import random

# Importamos los evaluadores oficiales de IFEval
from lm_eval.tasks.ifeval.utils import test_instruction_following_strict, InputExample


class TareaIFEval(TareaBase):
    """
    Evaluador para IFEval.
    Mide si el modelo cumple instrucciones verificables.
    """

    def __init__(self):
        self.datos_reales = []

    def cargar_datos(self, max_ejemplos: int = 150):
        print("Descargando dataset IFEval...")

        dataset = load_dataset("google/IFEval", split="train")

        seed = 0
        random.seed(seed)

        items = list(dataset)
        if len(items) > max_ejemplos:
            items = random.sample(items, max_ejemplos)
        self.datos_reales = []

        for i, item in enumerate(items):
            item_dict = dict(item)
            item_dict["task_id"] = f"IFEVAL/{i}"
            self.datos_reales.append(item_dict)
        return self.datos_reales

    def construir_prompt(self, item):
        instruccion_sistema = (
            "We are evaluating your capability of following instructions"
            "Follow the user's instructions exactly"
        )

        prompt_final = [
            {"role": "system", "content": instruccion_sistema},
            {"role": "user", "content": item["prompt"]}
        ]

        return prompt_final

    def evaluar(self, predicciones, nombre_modelo):
        aciertos_prompt = 0
        total = len(self.datos_reales)

        resultados_detalle = []

        for pred, real in zip(predicciones, self.datos_reales):
            respuesta_ia = pred["completion"]

            ejemplo_eval = InputExample(
                key=real["key"],
                instruction_id_list=real["instruction_id_list"],
                prompt=real["prompt"],
                kwargs=real["kwargs"],
            )

            resultado = test_instruction_following_strict(ejemplo_eval, respuesta_ia)

            if resultado.follow_all_instructions:
                aciertos_prompt += 1

            resultados_detalle.append(resultado)

        nota = (aciertos_prompt / total) * 100

        print(f"Precisión del modelo '{nombre_modelo}' en IFEval: {nota:.2f}%")
        return nota
    
