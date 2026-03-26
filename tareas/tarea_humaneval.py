from tareas.tarea_base import TareaBase
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

class TareaHumanEval(TareaBase):
    """
    Evaluador de código Python usando la librería oficial de OpenAI (HumanEval).
    """

    def cargar_datos(self):
        # 1. Tu lógica exacta para leer los problemas
        problemas = read_problems()
        return list(problemas.values()) # HumanEval devuelve un diccionario, lo convertimos a lista

    def construir_prompt(self, item):
        prompt = item['prompt']
        instruccion = (
            "You are an expert Python programmer. Complete the following Python function. "
            "Respond ONLY with valid Python code. "
            "Do NOT write explanations, do NOT write tests, do NOT use markdown (```) "
            "and do NOT write an if __name__ == '__main__' block.\n\n"
        )
        return instruccion + prompt

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