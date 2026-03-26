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
    print("="*50)
    
    # 2. LAS FÁBRICAS NOS DAN LAS PIEZAS EXACTAS PARA ESTE EXPERIMENTO
    motor = crear_motor(config)
    tarea = crear_tarea(config)
    
    # 3. Preparamos el examen
    problemas = tarea.cargar_datos()
    predicciones = []
    n_tokens_totales = 0
    tiempo_inferencia_total = 0.0
    
    # 4. Activamos codecarbon
    print("\nIniciando medición de energía...")
    tracker = EmissionsTracker(log_level="error") # log_level="error" quita los warnings irrelevantes
    tracker.start()
    
    # 5. EL BUCLE UNIVERSAL DE GENERACIÓN
    for problema in problemas:
        # El benchmark adapta la pregunta
        id_problema = problema["task_id"]
        prompt = tarea.construir_prompt(problema)
        print(f"\nGenerando respuesta para el problema {id_problema}... Prompt: {prompt}...")
        # El motor responde y medimos su tiempo
        inicio = time.time()
        resultado = motor.generar_respuesta(prompt, config.max_tokens)
        tiempo_inferencia_total += (time.time() - inicio)
        
        # Guardamos lo que ha respondido y los tokens que ha gastado
        predicciones.append({
            "task_id": id_problema,
            "completion": resultado["texto"]
        })
        n_tokens_totales += resultado["tokens_generados"]

    # 6. APAGAMOS CODECARBON
    tracker.stop()
    print("Medición de energía finalizada.")
    
    energia_kwh = tracker.final_emissions_data.energy_consumed
    co2_kg = tracker.final_emissions_data.emissions
    
    # 7. CORREGIMOS EL EXAMEN (Fuera del medidor de energía)
    print(f"\nEvaluando resultados con la métrica de {config.tarea.upper()}...")
    precision = tarea.evaluar(predicciones, config.nombre_modelo)
    
    # 8. EMPAQUETAMOS LOS RESULTADOS FINALES
    return Metricas(energia_kwh, co2_kg, n_tokens_totales, tiempo_inferencia_total, precision)


if __name__ == "__main__":
    #gpu_actual = get_gpu_name()     
    configuracion_actual = ConfigExperimento(
        nombre_modelo="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hardware="cpu",
        nombre_hardware="Procesador	11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz, 2419 Mhz, 4 procesadores principales, 8 procesadores lógicos",             
        motor="hf",                 
        tarea="humaneval",
        max_tokens=256                    
    )
    
    resultados_finales = ejecutar_medicion(configuracion_actual)
    
    print("\n" + "*"*50)
    print("RESULTADOS FINALES DEL EXPERIMENTO")
    print("*"*50)
    resultados_finales.imprimir_metricas()
    path = "resultados2.csv"
    resultados_finales.guardar_csv(configuracion_actual, path)