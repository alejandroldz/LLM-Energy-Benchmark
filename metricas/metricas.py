from configuraciones.experimentos import ConfigExperimento
import pandas as pd
import os

class Metricas:
    def __init__(self, energia_kwh, co2_kg, tokens_prompt, tokens_gen, ttft_total, tiempo_inferencia, num_problemas, precision):
        self.energia_total_kwh = energia_kwh
        self.co2_total_kg = co2_kg
        self.tokens_prompt = tokens_prompt
        self.tokens_gen = tokens_gen
        self.ttft_total = ttft_total
        self.tiempo_inferencia = tiempo_inferencia
        self.num_problemas = num_problemas
        self.precision = precision
    
    def _safe_div(self, numerador, denominador):
        if denominador == 0:
            return 0.0
        return numerador / denominador

    def julios_totales(self):
        return self.energia_total_kwh * 3600000
    
    def julios_por_token_gen(self):
        return self._safe_div(self.julios_totales(), self.tokens_gen)
    
    def tiempo_decodificacion(self):
        # El tiempo puro generando palabras (quitando el tiempo de prefill)
        return self.tiempo_inferencia - self.ttft_total

    def tokens_por_segundo_decodificacion(self):
        # Tokens/s puros de escritura
        return self._safe_div(self.tokens_gen, self.tiempo_decodificacion())
    
    def latencia_por_token(self):
        return self._safe_div(self.tiempo_decodificacion(), self.tokens_gen)
        
    def ttft_medio(self):
        return self.ttft_total/self.num_problemas 
    
    def edp(self):
        return self.julios_totales() * self.tiempo_inferencia
        
    def edp_por_token(self):
        return self.julios_por_token_gen() * self.latencia_por_token()
    
    def imprimir_metricas(self):
        print(f"Energía total consumida: {self.energia_total_kwh:.6f} kWh")
        print(f"Emisiones de CO2 estimadas: {self.co2_total_kg:.6f} kg")
        print(f"Tokens de Prompt (Prefill): {self.tokens_prompt}")
        print(f"Tokens Generados (Escritura): {self.tokens_gen}")
        print(f"Tiempo TTFT Total (Prefill): {self.ttft_total:.4f} s")
        print(f"Tiempo Decodificación (Decode): {self.tiempo_decodificacion():.4f} s")
        print(f"Tiempo Total Inferencia: {self.tiempo_inferencia:.4f} s")
        print(f"Precisión: {self.precision:.4f} %")
        print(f"Energía por token generado: {self.julios_por_token_gen():.4f} J/token")
        print(f"Velocidad de generación: {self.tokens_por_segundo_decode():.4f} tokens/s")
        print(f"Latencia por token: {self.latencia_por_token():.4f} s/token")
        print(f"Energy-Delay Product (EDP): {self.edp():.4f} J*s")

    def guardar_csv(self, config: ConfigExperimento, path: str):
        df = pd.DataFrame({
            "Modelo": [config.nombre_modelo],
            "Motor": [config.motor],
            "Hardware": [config.hardware],
            "Tarea": [config.tarea],
            "Energía total consumida": [self.energia_total_kwh],
            "Emisiones de CO2": [self.co2_total_kg],
            "Tokens Prompt": [self.tokens_prompt],
            "Tokens Generados": [self.tokens_gen],
            "TTFT Total": [self.ttft_total],
            "TTFT medio": [self.ttft_medio()],
            "Tiempo Inferencia Total": [self.tiempo_inferencia],
            "Número de problemas": [self.num_problemas],
            "Precisión": [self.precision],
            "Energía J/token gen": [self.julios_por_token_gen()],
            "Tokens/s (Decode)": [self.tokens_por_segundo_decodificacion()],
            "Latencia s/token": [self.latencia_por_token()],
            "EDP": [self.edp()]
        })

        archivo_existe = os.path.exists(path)
        df.to_csv(path, mode='a', header=not archivo_existe, index=False)