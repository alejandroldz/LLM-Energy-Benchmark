from configuraciones.experimentos import ConfigExperimento
from tareas.tarea_humaneval import TareaHumanEval
from tareas.tarea_mmlu import TareaMMLU
from tareas.tarea_resumen import TareaResumen
from tareas.tarea_ifeval import TareaIFEval
def crear_tarea(config: ConfigExperimento):
    """
    Fábrica que decide qué benchmark instanciar.
    """
    if config.tarea == "humaneval":
        return TareaHumanEval()

    elif config.tarea == "mmlu":
        return TareaMMLU()

    elif config.tarea == "resumen":
        return TareaResumen()
    elif config.tarea == "ifeval":
        return TareaIFEval()

    else:
        raise ValueError(f"Tarea desconocida: {config.tarea}")