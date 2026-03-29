import time
from codecarbon import EmissionsTracker
from check_hardware import *
from configuraciones.experimentos import ConfigExperimento
from motores.factory import crear_motor
from tareas.factory import crear_tarea
from metricas.metricas import Metricas

def ejecutar_medicion(config: ConfigExperimento) -> Metricas:
    print("\n" + "="*50)
    print(f"INICIANDO EXPERIMENTO")
    print(f"Modelo:  {config.nombre_modelo}")
    print(f"Motor:   {config.motor.upper()} | Hardware: {config.hardware.upper()}")
    print(f"Tarea:   {config.tarea.upper()}")
    print(f"Batch:   {config.batch_size}")
    print("="*50)
    
    motor = crear_motor(config)
    tarea = crear_tarea(config)
    
    # 3. Preparamos el examen
    problemas = tarea.cargar_datos()
    predicciones = []
    n_tokens_generados = 0
    n_tokens_prompt = 0
    ttft_total = 0
    tiempo_inferencia_total = 0.0
    num_problemas = len(problemas)
    # 4. Activamos codecarbon
    print("Calentando el modelo (Warm-up)...")
    print(motor.generar_respuesta(["Hola, dime tu nombre de modelo e información extra"], max_tokens=100)[0]["texto"])
    
    print("\nIniciando medición de energía...")
    tracker = EmissionsTracker(log_level="error")
    tracker.start()
    
    # 5. EL BUCLE UNIVERSAL DE GENERACIÓN 
    for i in range(0, num_problemas, config.batch_size):
        # Extraemos el lote actual de problemas
        lote_problemas = problemas[i : i + config.batch_size]
        # Construimos todos los prompts de este lote de golpe
        lote_prompts = [tarea.construir_prompt(p) for p in lote_problemas]
        
        num_lote = (i // config.batch_size) + 1
        print(f"\nProcesando lote {num_lote}... (Problemas {i} a {i + len(lote_problemas) - 1})")
        
        # El motor responde al lote entero y medimos su tiempo
        inicio = time.time()
        resultados_lote = motor.generar_respuesta(lote_prompts, config.max_tokens)
        tiempo_inferencia_total += (time.time() - inicio)
        
        # Guardamos lo que ha respondido y los tokens que ha gastado emparejando listas
        for problema, resultado in zip(lote_problemas, resultados_lote):
            predicciones.append({
                "task_id": problema["task_id"],
                "completion": resultado["texto"]
            })
            n_tokens_prompt += resultado["tokens_prompt"]
            n_tokens_generados += resultado["tokens_generados"]
            ttft_total += resultado["ttft"]

    # Paramos codecarbon
    tracker.stop()
    print("Medición de energía finalizada.")
    
    energia_kwh = tracker.final_emissions_data.energy_consumed
    co2_kg = tracker.final_emissions_data.emissions
    
    # Corregimos
    print(f"\nEvaluando resultados con la métrica de {config.tarea.upper()}...")
    precision = tarea.evaluar(predicciones, config.nombre_modelo)
    
    # Devolvemos el objeto de métricas con toda la información recopilada
    return Metricas(energia_kwh, co2_kg, n_tokens_prompt, n_tokens_generados, tiempo_inferencia_total, ttft_total, num_problemas, precision)


if __name__ == "__main__":
    #gpu_actual = get_gpu_name()     
    configuracion_actual = ConfigExperimento(
        nombre_modelo="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        archivo_gguf="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        hardware="cpu", 
        nombre_hardware="Procesador 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz, 2419 Mhz, 4 procesadores principales, 8 procesadores lógicos",            
        motor="hf",                    
        tarea="humaneval",
        max_tokens=256,
        batch_size=1                    
    )
    
    resultados_finales = ejecutar_medicion(configuracion_actual)
    
    print("\n" + "*"*50)
    print("RESULTADOS FINALES DEL EXPERIMENTO")
    print("*"*50)
    resultados_finales.imprimir_metricas()
    path = "resultados.csv"
    resultados_finales.guardar_csv(configuracion_actual, path)