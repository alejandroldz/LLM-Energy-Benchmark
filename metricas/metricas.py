import pandas as pd
import os
from configuraciones.experimentos import ConfigExperimento 

class Metricas:
    def __init__(self,energia_kwh,co2_kg,tokens_prompt,tokens_gen,tiempo_inferencia,ttft_total,num_problemas,precision,num_lotes=None,):
        self.energia_total_kwh = energia_kwh
        self.co2_total_kg = co2_kg
        self.tokens_prompt = tokens_prompt
        self.tokens_gen = tokens_gen
        self.tiempo_inferencia = tiempo_inferencia
        self.ttft_total = ttft_total
        self.num_problemas = num_problemas
        self.precision = precision
        self.num_lotes = num_lotes if (num_lotes is not None and num_lotes > 0) else num_problemas
    
    def _safe_div(self, numerador, denominador):
        return numerador / denominador if denominador > 0 else 0.0

    def julios_totales(self):
        return self.energia_total_kwh * 3600000
    
    def julios_por_token_gen(self):
        # Energía gastada por cada token que produce el modelo
        return self._safe_div(self.julios_totales(), self.tokens_gen)

    def julios_por_token_total(self):
        # Energía distribuida entre el contexto procesado y el generado
        return self._safe_div(self.julios_totales(), (self.tokens_prompt + self.tokens_gen))

    def tiempo_decodificacion_total(self):
        # Tiempo inferencia total menos el tiempo gastado en el TTFT
        return max(0.0, self.tiempo_inferencia - self.ttft_total)


    def tpot(self):
        # Time Per Output Token: La latencia pura de escribir (s/token)
        return self._safe_div(self.tiempo_decodificacion_total(), self.tokens_gen)

    def throughput_decode(self):
        # Tokens/s en fase de escritura
        return self._safe_div(self.tokens_gen, self.tiempo_decodificacion_total())
    
    def throughput_total(self):
        # Tokens/s globales (incluye lo rápido que se procesó el prompt)
        return self._safe_div((self.tokens_prompt + self.tokens_gen), self.tiempo_inferencia)

    def ttft_medio(self):
        return self._safe_div(self.ttft_total, self.num_lotes)
        
    def tiempo_inferencia_medio(self):
        return self._safe_div(self.tiempo_inferencia, self.num_problemas)

    def edp(self):
        # Energy-Delay Product: Menor es mejor (equilibrio entre rapidez y bajo consumo)
        return self.julios_totales() * self.tiempo_inferencia
        
    def edp_por_token(self):
        return self.julios_por_token_gen() * self.tpot()


    def imprimir_metricas(self):
        print("--- MÉTRICAS DEL EXPERIMENTO ---")
        print(f"Energía y CO2:")
        print(f"  -Energía total: {self.energia_total_kwh:.6f} kWh ({self.julios_totales():.2f} J)")
        print(f"  -Emisiones CO2: {self.co2_total_kg:.6f} kg")
        print(f"  -Energía / Token Generado: {self.julios_por_token_gen():.4f} J")
        print(f"  -EDP Total: {self.edp():.4f} J*s")
        
        print(f"\nTrabajo del Modelo:")
        print(f"  -Tareas (Prompts): {self.num_problemas} | Lotes: {self.num_lotes} | Precisión: {self.precision:.2f}%")
        print(f"  -Tokens Contexto (Prefill): {self.tokens_prompt}")
        print(f"  -Tokens Generados (Decode): {self.tokens_gen}")
        
        print(f"\nTiempos y Latencias (Promedio por lote):")
        print(f"  - TTFT (Latencia inicial): {self.ttft_medio():.4f} s")
        print(f"  - Tiempo Inferencia Total: {self.tiempo_inferencia_medio():.4f} s")
        
        print(f"\nVelocidad (Throughput):")
        print(f"  - TPOT (Latencia de escritura): {self.tpot():.4f} s/token")
        print(f"  - Velocidad Decode: {self.throughput_decode():.2f} tokens/s")
        print(f"  - Velocidad Total (Prefill+Decode): {self.throughput_total():.2f} tokens/s")
        print("--------------------------------")

    def guardar_csv(self, config: ConfigExperimento, path: str):    
        df = pd.DataFrame({
            "Modelo": [config.nombre_modelo],
            "Motor": [config.motor],
            "Hardware": [config.hardware],
            "Tarea": [config.tarea],
            "Numero de Problemas": [self.num_problemas],
            "Numero de Lotes": [self.num_lotes],
            "Tamaño Batch": [config.batch_size],
            "Precisión": [self.precision],
            "Energía (kWh)": [self.energia_total_kwh],
            "Energía (J)": [self.julios_totales()],
            "CO2 (kg)": [self.co2_total_kg],
            "EDP (J*s)": [self.edp()],
            "Tokens Prompt": [self.tokens_prompt],
            "Tokens Generados": [self.tokens_gen],
            "TTFT Medio (s)": [self.ttft_medio()],
            "TPOT (s/token)": [self.tpot()],
            "Throughput Decode (tok/s)": [self.throughput_decode()],
            "Throughput Total (tok/s)": [self.throughput_total()],
            "Inferencia Media (s)": [self.tiempo_inferencia_medio()],
            "Energía (J/token gen)": [self.julios_por_token_gen()],
            "Cuantizacon": [config.cuantizacion]
        })

        archivo_existe = os.path.exists(path)
        df.to_csv(path, mode='a', header=not archivo_existe, index=False)
