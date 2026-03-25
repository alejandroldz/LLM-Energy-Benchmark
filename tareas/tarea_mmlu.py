from tareas.tarea_base import TareaBase
from datasets import load_dataset

class TareaMMLU(TareaBase):
    """
    Evaluador para el benchmark MMLU (Massive Multitask Language Understanding).
    Adaptado a nuestra arquitectura modular para mantener el control de los tokens.
    """
    def __init__(self):
        # Aquí guardaremos las respuestas correctas para corregir el examen luego
        self.datos_reales = []

    def cargar_datos(self):
        # MMLU es gigante. Para empezar, cargamos una sub-tarea (ej: álgebra).
        print("Descargando dataset MMLU...")
        dataset = load_dataset("cais/mmlu", "all")
        # Guardamos los datos en la memoria de la clase
        self.datos_reales = []
        for i, item in enumerate(dataset):
            item_dict = dict(item)
            item_dict["task_id"] = f"MMLU/{i}"
            self.datos_reales.append(item_dict)

        return self.datos_reales

    def construir_prompt(self, item):
        # MMLU es un examen tipo test. Construimos la pregunta.
        pregunta = item['question']
        opciones = item['choices']
        
        prompt = f"Question: {pregunta}\n"
        letras = ['A', 'B', 'C', 'D']
        
        for letra, opcion in zip(letras, opciones):
            prompt += f"{letra}. {opcion}\n"
            
        prompt += "Answer:"
        return prompt

    def evaluar(self, predicciones, nombre_modelo):
        aciertos = 0
        total = len(self.datos_reales)
        opciones = ['A', 'B', 'C', 'D']
        for pred, real in zip(predicciones, self.datos_reales):
            # Limpiamos espacios y pasamos a mayúsculas lo que respondió la IA
            respuesta_ia = pred["completion"].strip().upper()
            
            # La respuesta correcta viene como un número (0=A, 1=B, 2=C, 3=D)
            indice_correcto = real['answer']
            letra_correcta = opciones[indice_correcto]

            # Comprobamos si la primera letra que escupió el modelo es la correcta
            if respuesta_ia.startswith(letra_correcta):
                aciertos += 1

        nota = (aciertos / total) * 100
        print(f"Precisión del modelo '{nombre_modelo}' en MMLU: {nota:.2f}%")
        return nota