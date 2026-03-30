from human_eval.evaluation import evaluate_functional_correctness
import os

def evaluar_archivo():
    # El nombre exacto de tu archivo generado
    archivo_muestras = "muestras_Qwen_Qwen2.5-Coder-3B-Instruct.jsonl"
    
    # Comprobamos que el archivo existe donde toca
    if not os.path.exists(archivo_muestras):
        print(f"❌ ERROR: No se encuentra el archivo '{archivo_muestras}'.")
        print("Asegúrate de ejecutar este script en la misma carpeta donde está el .jsonl")
        return

    print("\n" + "="*50)
    print(f"🚀 INICIANDO EVALUACIÓN DE HUMANEVAL")
    print(f"📂 Archivo: {archivo_muestras}")
    print("="*50)
    print("\nEjecutando los test unitarios (esto puede tardar un par de minutos)...")

    # Lanzamos el evaluador oficial
    # Nota: n_workers define cuántos problemas se evalúan en paralelo. 
    # En Windows, si te da errores extraños de multiprocesamiento, bájalo a n_workers=1
    resultados = evaluate_functional_correctness(
        sample_file=archivo_muestras,
        k=[1],          # Queremos la métrica pass@1 (acertar al primer intento)
        n_workers=4,    # Hilos paralelos para ir más rápido
        timeout=3.0     # Tiempo límite por test (ignorado si hiciste el hack de Windows)
    )

    # Extraemos la nota y la pasamos a porcentaje
    nota_pass_1 = resultados['pass@1'] * 100

    print("\n" + "🌟"*25)
    print(" "*10 + "RESULTADO FINAL")
    print("🌟"*25)
    print(f"🎯 Precisión (Pass@1): {nota_pass_1:.2f}%")
    print("🌟"*25 + "\n")

if __name__ == "__main__":
    evaluar_archivo()