from tareas.tarea_base import TareaBase
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

class TareaHumanEval(TareaBase):
    """
    Evaluador de código Python usando la librería oficial de OpenAI (HumanEval).
    """

    def cargar_datos(self):
        problemas = read_problems()
        return list(problemas.values()) # HumanEval devuelve un diccionario, lo convertimos a lista

    def construir_prompt(self, item):
        prompt_codigo = "Complete the following function body. DO NOT write any additional text, just cotinue the given code. Do NOT add ```python at the beginning or end of the code:" + item['prompt']
        
        instruccion_sistema = (
            "You are a strict Python code completion engine. "
            "Complete the following function body without any explanations, apologies, or additional text, just the code.\n\n"
            "Only continue the given function body.\n\n"
            "Do NOT add ```python at the beginning or end of the code."
        )
        
        prompt_final = [
            {"role": "system", "content": instruccion_sistema},
            {"role": "user", "content": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n"},
            {"role": "assistant", "content": "    return a + b\n"},
            {"role": "user", "content": prompt_codigo},
            {"role": "assistant", "content": "    "}
        ]
        
        return prompt_final

    def evaluar(self, codigos, nombre_modelo):
        nombre_archivo = f"muestras_{nombre_modelo.replace('/', '_')}.jsonl"
        write_jsonl(nombre_archivo, codigos)
        
        resultados = evaluate_functional_correctness(
            sample_file=nombre_archivo, # nombre del archivo con las muestras
            k=[1],                      # queremos ver si acierta a la primera
            n_workers=4,                # 4 procesos en paralelo
            timeout=10                  # para evitar bucles infinitos
        )

        nota = resultados['pass@1'] * 100
        print(f"Precisión del modelo '{nombre_modelo}': {nota:.2f}%")
        return nota