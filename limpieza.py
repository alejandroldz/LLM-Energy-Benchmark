import json

def limpiar_jsonl_humaneval(ruta_entrada: str, ruta_salida: str):
    """
    Lee un archivo JSONL de HumanEval, recorta la basura de las respuestas
    y guarda un nuevo archivo limpio listo para evaluar.
    """
    # Lista exhaustiva de "Stop Words" basada en el comportamiento de TinyLlama
    stop_words = [
        "\nif __name__",
        "\n# Test",
        "\n# test",
        "\n# Your code here", # A veces lo pone antes de la explicación
        "\nassert ",
        "\nExplanation",
        "\nprint(",
        "```",               # Cierra el bloque de código markdown
        "\nExample:"
    ]

    lineas_procesadas = 0
    lineas_recortadas = 0

    with open(ruta_entrada, 'r', encoding='utf-8') as f_in, \
         open(ruta_salida, 'w', encoding='utf-8') as f_out:
        
        for linea in f_in:
            datos = json.loads(linea)
            codigo_original = datos["completion"]
            
            # Inicializamos el punto de corte al final del texto por defecto
            posicion_corte = len(codigo_original)
            
            # Buscamos cuál es la PRIMERA stop_word que aparece en el texto
            for palabra in stop_words:
                indice = codigo_original.find(palabra)
                
                # Si la palabra existe y aparece ANTES que nuestro corte actual, actualizamos la tijera
                if indice != -1 and indice < posicion_corte:
                    posicion_corte = indice
            
            # Aplicamos el tijeretazo si hemos encontrado algo
            if posicion_corte < len(codigo_original):
                codigo_limpio = codigo_original[:posicion_corte]
                lineas_recortadas += 1
            else:
                codigo_limpio = codigo_original
            
            # Sobrescribimos el campo y lo guardamos en el nuevo archivo
            datos["completion"] =" " + codigo_limpio
            f_out.write(json.dumps(datos) + '\n')
            lineas_procesadas += 1

    print("\nProceso de limpieza terminado.")
    print(f"Problemas procesados: {lineas_procesadas}")
    print(f"Problemas donde la IA alucinó y se le recortó basura: {lineas_recortadas} ({(lineas_recortadas/lineas_procesadas)*100:.1f}%)")
    print(f"Archivo limpio guardado en: {ruta_salida}\n")

if __name__ == '__main__':
    archivo_sucio = "muestras_TinyLlama_TinyLlama-1.1B-Chat-v1.0.jsonl"
    archivo_limpio = "limpio_muestras_TinyLlama.jsonl"
    
    limpiar_jsonl_humaneval(archivo_sucio, archivo_limpio)