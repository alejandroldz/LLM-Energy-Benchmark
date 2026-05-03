from tareas.tarea_base import TareaBase
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
import json
import ast
import re
class TareaHumanEval(TareaBase):
    """
    Evaluador de código Python usando la librería oficial de OpenAI (HumanEval).
    """

    def cargar_datos(self):
        problemas = read_problems()
        return list(problemas.values()) # HumanEval devuelve un diccionario, lo convertimos a lista

    def construir_prompt(self, item):
        prompt_codigo = "Complete the following function body. DO NOT write any additional text, just cotinue the given code. Do NOT add ```python at the beginning or end of the code. Do NOT write in markdown." + item['prompt']
        
        instruccion_sistema = (
            "You are a strict Python code completion engine. "
            "Complete the following function body without any explanations, apologies, or additional text, just the code.\n\n"
            "Only continue the given function body.\n\n"
            "Do NOT add ```python at the beginning or  ``` at the end of the code."
            "Do NOT add comments like # comments or \"\"\" comments.\n\n"
            "It is not allowed to write in markdown.\n\n"

        )
        
        prompt_final = [
            {"role": "system", "content": instruccion_sistema},
            {"role": "user", "content": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n"},
            {"role": "assistant", "content": "    return a + b\n"},
            {"role": "user", "content": "Complete the following function body. Do not rewrite any additional text, just continue the given code: " + prompt_codigo},
            {"role": "assistant", "content": "  "}
        ]
        
        return prompt_final

    def _extraer_cuerpo_humaneval(self, texto_generado: str) -> str:
        texto = texto_generado
        
        # Quitar markdown de forma segura 
        match = re.search(r'```(?:python)?(.*?)```', texto, re.DOTALL | re.IGNORECASE)
        if match:
            texto = match.group(1)
        else:
            texto = texto.replace('```' + 'python', '').replace('```', '')

        lineas = texto.split('\n')
        lineas_validas = []
    
        for linea in lineas:
            linea_strip = linea.strip()
            
            if not lineas_validas and not linea_strip:
                continue
                
            if linea_strip.startswith('def ') or linea_strip.startswith('import ') or linea_strip.startswith('from '):
                continue
                
            indentacion = len(linea) - len(linea.lstrip())
            if not lineas_validas and indentacion == 0 and not linea_strip.startswith('#'):
                continue
                
            if lineas_validas and indentacion == 0 and linea_strip and not linea_strip.startswith('#'):
                break
                
            lineas_validas.append(linea)
            
            if not lineas_validas:
                return "    pass\n"
            
        return "\n".join(lineas_validas)

    def evaluar(self, codigos, nombre_modelo):
        codigos_limpios = []
        
        for item in codigos:
            # Limpiamos los codigos generados
            codigo_perfecto = self._extraer_cuerpo_humaneval(item["completion"])
            
            codigos_limpios.append({
                "task_id": item["task_id"],
                "completion": codigo_perfecto
            })

        nombre_archivo = f"muestras_{nombre_modelo.replace('/', '_')}.jsonl"
        write_jsonl(nombre_archivo, codigos_limpios)
        
        resultados = evaluate_functional_correctness(
            sample_file=nombre_archivo,
            k=[1],
            n_workers=1,
            timeout=20
        )

        nota = resultados['pass@1'] * 100
        print(f"Precisión del modelo '{nombre_modelo}': {nota:.2f}%")
        return nota