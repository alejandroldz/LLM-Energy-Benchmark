import json
import re
from human_eval.evaluation import evaluate_functional_correctness

def limpiar_codigo(texto_generado: str) -> str:
    # chr(96) es el acento grave. Multiplicado por 3 forma el marcador de markdown
    marcador = chr(96) * 3
    match = re.search(rf'{marcador}(?:python)?(.*?){marcador}', texto_generado, re.DOTALL | re.IGNORECASE)
    if match:
        texto = match.group(1)
    else:
        texto = texto_generado.replace(f'{marcador}python', '').replace(marcador, '')

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

def reevaluar_archivo(ruta_entrada: str, ruta_salida: str):
    codigos_limpios = []
    
    with open(ruta_entrada, 'r', encoding='utf-8') as f:
        for linea in f:
            item = json.loads(linea)
            codigo_perfecto = limpiar_codigo(item["completion"])
            codigos_limpios.append({
                "task_id": item["task_id"],
                "completion": "\n" + codigo_perfecto
            })
            
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        for item in codigos_limpios:
            f.write(json.dumps(item) + '\n')
            
    print(f"\n[+] Archivo limpiado y guardado en: {ruta_salida}")
    print("[+] Iniciando evaluación oficial de HumanEval...")
    print("[!] Usando n_workers=1 para evitar los errores de 'timed out'\n")
    
    resultados = evaluate_functional_correctness(
        sample_file=ruta_salida,
        k=[1],
        n_workers=1,
        timeout=10
    )
    
    nota = resultados['pass@1'] * 100
    print(f"======================================")
    print(f"NOTA FINAL REEVALUADA: {nota:.2f}%")
    print(f"======================================")

if __name__ == "__main__":
    # IMPORTANTE: Cambia esto por el nombre de tu archivo JSONL original
    ARCHIVO_ORIGINAL = "muestras_TinyLlama_TinyLlama-1.1B-Chat-v1.0.jsonl"
    ARCHIVO_LIMPIO = "muestasknekvnras_limpias_reevaluadasssss.jsonl"
    
    try:
        reevaluar_archivo(ARCHIVO_ORIGINAL, ARCHIVO_LIMPIO)
    except FileNotFoundError:
        print(f"ERROR: No se encuentra el archivo '{ARCHIVO_ORIGINAL}'.")
        print("Asegúrate de poner la ruta correcta en la variable ARCHIVO_ORIGINAL.")