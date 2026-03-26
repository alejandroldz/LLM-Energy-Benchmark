from configuraciones.experimentos import ConfigExperimento
import pandas as pd
class Metricas:
    def __init__(self, energia_kwh, co2_kg, tokens_gen, tiempo_inferencia, precision):
        self.energia_total_kwh = energia_kwh
        self.co2_total_kg = co2_kg
        self.tokens_gen = tokens_gen
        self.tiempo_inferencia = tiempo_inferencia
        self.precision = precision
    
    def _safe_div(self, numerador, denominador):
        if denominador == 0:
            return 0.0
        return numerador / denominador

    def julios_totales(self):
        return self.energia_total_kwh * 3600000
    
    def julios_por_token_gen(self):
        return self._safe_div(self.julios_totales(), self.tokens_gen)
    
    def tokens_por_segundo(self):
        return self._safe_div(self.tokens_gen, self.tiempo_inferencia)
    
    def latencia_por_token(self):
        return self._safe_div(self.tiempo_inferencia, self.tokens_gen)
    
    def edp(self):
        return self.julios_totales() * self.tiempo_inferencia
    def edp_por_token(self):
        return self.julios_por_token_gen() * self.latencia_por_token()
    
    def imprimir_metricas(self):
        print(f"Energía total consumida: {self.energia_total_kwh:.4f} kWh")
        print(f"Emisiones de CO2 estimadas: {self.co2_total_kg:.4f} kg")
        print(f"Tokens generados: {self.tokens_gen}")
        print(f"Tiempo total de inferencia: {self.tiempo_inferencia:.4f} segundos")
        print(f"Precisión: {self.precision:.4f} %")
        print(f"Energía por token generado: {self.julios_por_token_gen():.4f} J/token")
        print(f"Tokens por segundo: {self.tokens_por_segundo():.4f} tokens/s")
        print(f"Latencia por token: {self.latencia_por_token():.4f} s/token")
        print(f"Energy-Delay Product (EDP): {self.edp():.4f} J*s")
        print(f"EDP por token: {self.edp_por_token():.4f} J*s/token")

    def guardar_csv(self, config: ConfigExperimento, path: str):
        df = pd.DataFrame({
            "Modelo": [config.nombre_modelo],
            "Motor": [config.motor],
            "Hardware": [config.hardware],
            "Tarea": [config.tarea],
            "Energía total consumida": [self.energia_total_kwh],
            "Emisiones de CO2 estimadas": [self.co2_total_kg],
            "Tokens generados": [self.tokens_gen],
            "Tiempo total de inferencia": [self.tiempo_inferencia],
            "Precisión": [self.precision],
            "Energía por token generado": [self.julios_por_token_gen()],
            "Tokens por segundo": [self.tokens_por_segundo()],
            "Latencia por token": [self.latencia_por_token()],
            "Energy-Delay Product (EDP)": [self.edp()],
            "EDP por token": [self.edp_por_token()]
        })

        df.to_csv(path, index=False)
