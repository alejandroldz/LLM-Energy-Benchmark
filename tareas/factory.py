from configuraciones.experimentos import ConfigExperimento
from tareas.tarea_humaneval import TareaHumanEval
from tareas.tarea_mmlu import TareaMMLU  

def crear_tarea(config: ConfigExperimento):
    """
    Fábrica que decide qué benchmark (examen) instanciar.
    """
    if config.tarea == "humaneval":
        return TareaHumanEval()
    
    elif config.tarea == "mmlu":
        return TareaMMLU()
        
    else:
        raise ValueError(f"Tarea desconocida: {config.tarea}")