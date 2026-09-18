# ClaimScope: plan de implementación

Documento para Claude Code. Léelo entero antes de escribir código. Trabaja fase a fase y no avances a la siguiente hasta cumplir los criterios de aceptación de la actual.

## 1. Objetivo

Un agente construido con LangGraph que, dado un paper de ML (arXiv), extrae sus afirmaciones, decide cuáles son verificables con un presupuesto de cómputo limitado, diseña una versión reducida del experimento, la ejecuta en un sandbox y emite un veredicto estadísticamente fundamentado.

El proyecto NO pretende reproducir papers a escala completa. Pretende responder: "¿se sostiene esta afirmación relativa a escala reducida?".

### No objetivos

- Reproducir números absolutos de las tablas del paper.
- Soportar papers cuyo efecto depende de la escala (preentrenamiento de LLMs, capacidades emergentes, scaling laws grandes).
- Interfaz web en las primeras fases. La interfaz es una CLI.

## 2. Conceptos del dominio

### Tipos de afirmación

| Tipo | Ejemplo | ¿Verificable a escala reducida? |
|---|---|---|
| `absolute` | "Obtenemos 84.3% top-1 en ImageNet" | No. Se registra y se descarta. |
| `comparative` | "El método A supera al baseline B" | Sí. Objetivo principal. |
| `ablation` | "Quitar el componente X empeora el resultado" | Sí. |
| `scaling_trend` | "La mejora crece con el tamaño del modelo" | Parcialmente, con 3 o 4 puntos pequeños. |

### Veredictos

- `consistent_at_reduced_scale`
- `not_consistent_at_reduced_scale`
- `inconclusive`
- `not_testable` (asignado en triage)

Regla de redacción del informe: un resultado negativo a escala reducida NO refuta el paper. El informe siempre lo indica.

### Reglas de reducción (invariantes)

1. Ambos brazos de una comparación reciben exactamente el mismo presupuesto, dataset, reducción y número de semillas.
2. Mínimo 3 semillas por brazo.
3. Se conserva la arquitectura; se reduce profundidad, anchura, datos o pasos.
4. Todo cambio respecto al paper se documenta en el plan de reducción con su justificación.

## 3. Stack

- Python 3.11+, gestión con `uv`.
- `langgraph` y `langchain-core`. Modelo vía `langchain-anthropic`, con el nombre del modelo configurable (por defecto `claude-sonnet-5`).
- `langgraph-checkpoint-sqlite` para persistencia y reanudación tras interrupts.
- `pydantic` v2 para todos los esquemas y salidas estructuradas del LLM.
- `arxiv` (paquete) y `pymupdf` para ingesta de PDFs.
- `docker` (SDK de Python) para el sandbox.
- `scipy` y `numpy` para estadística.
- `typer` y `rich` para la CLI.
- Trazas con Langfuse (self-hosted o cloud) o LangSmith, detrás de una variable de entorno. Debe funcionar sin trazas.
- `pytest`, `ruff`, `mypy`.

**Importante:** la API de LangGraph cambia con frecuencia (`interrupt`, `Command`, checkpointers). Antes de implementar el grafo, consulta la documentación actual de la versión instalada en lugar de asumir la API de memoria.

## 4. Estructura del repositorio

```
claimscope/
  pyproject.toml
  README.md
  CLAUDE.md                 # reglas persistentes para Claude Code (ver sección 11)
  .env.example
  src/claimscope/
    config.py               # settings con pydantic-settings
    state.py                # esquema del estado del grafo
    schemas.py              # Claim, ReductionPlan, RunResult, Verdict...
    graph.py                # construcción del grafo
    nodes/
      ingest.py
      extract_claims.py
      triage.py
      design_plan.py
      review.py             # interrupt de aprobación humana
      codegen.py
      execute.py
      debug.py
      analyze.py
      report.py
    sandbox/
      docker_runner.py
      images/Dockerfile.cpu
    stats.py
    prompts/                # prompts en ficheros .md versionados
    cli.py
  eval/
    annotations/            # papers anotados a mano (JSON)
    run_eval.py
    metrics.py
  tests/
  runs/                     # salidas por ejecución (ignorado en git)
```

## 5. Estado del grafo

```python
class Claim(BaseModel):
    id: str
    text: str                     # parafraseada, con referencia a sección/tabla
    source_location: str          # p. ej. "Table 2", "Sec. 4.1"
    claim_type: Literal["absolute", "comparative", "ablation", "scaling_trend"]
    arms: list[str]               # p. ej. ["method_A", "baseline_B"]
    metric: str
    expected_direction: str       # "A > B", "decreases without X"...
    testable: bool | None = None
    triage_reason: str | None = None

class ReductionPlan(BaseModel):
    claim_id: str
    original_setup: str
    reduced_setup: str
    changes: list[str]            # cada cambio con justificación
    preserved: list[str]
    why_claim_should_transfer: str
    seeds: int = 3
    estimated_minutes: float
    code_source: Literal["official_repo", "from_scratch"]

class RunResult(BaseModel):
    arm: str
    seed: int
    metric_value: float
    runtime_s: float
    log_path: str

class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: Literal[...]         # ver sección 2
    effect_estimate: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    notes: str

class GraphState(TypedDict):
    paper_id: str
    paper_text: str
    repo_url: str | None
    claims: list[Claim]
    selected_claim_ids: list[str]
    plans: dict[str, ReductionPlan]
    approved_plan_ids: list[str]
    workspace_dir: str
    run_results: dict[str, list[RunResult]]
    debug_attempts: dict[str, int]
    budget_minutes_total: float
    budget_minutes_used: float
    verdicts: list[ClaimVerdict]
    report_path: str | None
    errors: list[str]
```

## 6. Nodos y flujo

```
ingest -> extract_claims -> triage -> design_plan -> review (interrupt)
   review --aprobado--> codegen -> execute
   review --rechazado con feedback--> design_plan
   execute --error--> debug -> execute   (máx. N intentos por claim)
   execute --agotado N--> analyze (claim marcada inconclusive con motivo)
   execute --ok--> analyze -> report -> END
```

### Contratos por nodo

- **ingest**: descarga PDF de arXiv por ID, extrae texto con PyMuPDF, detecta enlace a repo oficial (GitHub) si existe. Cachea en `runs/<paper_id>/`.
- **extract_claims**: salida estructurada `list[Claim]`. Paráfrasis, nunca bloques largos copiados del paper. Siempre `source_location`.
- **triage**: marca `testable` y `triage_reason`. `absolute` es siempre no verificable. Rechaza claims cuyo efecto dependa de la escala o cuyo coste estimado supere el presupuesto. Selecciona como máximo K claims (configurable, por defecto 2).
- **design_plan**: genera `ReductionPlan` respetando las invariantes de la sección 2. Si hay repo oficial, prioriza `official_repo`.
- **review**: `interrupt` de LangGraph que muestra el plan en la CLI. El usuario aprueba, rechaza o da feedback en texto. El feedback vuelve a `design_plan`.
- **codegen**: si `official_repo`, clona y adapta configs y tamaños; si `from_scratch`, genera un script mínimo. En ambos casos produce un único punto de entrada `run.py --arm <arm> --seed <seed>` que escribe un JSON con la métrica.
- **execute**: primero un dry run corto (p. ej. 20-50 pasos) para medir tiempo real y extrapolar. Si la extrapolación supera el presupuesto restante, vuelve a `design_plan` con ese dato. Si cabe, ejecuta todos los brazos y semillas.
- **debug**: recibe traceback y código, propone parche. Límite de intentos configurable (por defecto 3).
- **analyze**: ver sección 8.
- **report**: Markdown en `runs/<paper_id>/report.md` con claims, triage, plan, resultados, veredictos y limitaciones.

## 7. Sandbox

- Contenedor Docker, usuario no root, límites de CPU, memoria y tiempo.
- Instalación de dependencias en una fase separada; la ejecución del experimento corre sin red.
- Solo se monta el directorio de trabajo del claim.
- Abstracción `Runner` con implementación `DockerCPURunner`. Dejar la interfaz preparada para un `GPURunner` futuro (GPU local o remota), sin implementarlo aún.
- El hardware de desarrollo no tiene CUDA: las fases 1 a 5 deben funcionar en CPU con experimentos de pocos minutos.

## 8. Estadística y regla de veredicto

- Por brazo: media y desviación típica sobre semillas.
- Efecto = diferencia de medias en la dirección esperada.
- Intervalo de confianza por bootstrap (sobre semillas) y test de Welch como apoyo.
- Regla:
  - IC entero en la dirección esperada -> `consistent_at_reduced_scale`
  - IC entero en la dirección opuesta -> `not_consistent_at_reduced_scale`
  - IC cruza cero -> `inconclusive`
- Con 3 semillas la potencia es baja: el informe debe decirlo explícitamente. Si sobra presupuesto, el agente puede proponer más semillas en lugar de más pasos.

## 9. Evaluación del propio agente

Formato de anotación (`eval/annotations/<paper_id>.json`):

```json
{
  "paper_id": "xxxx.xxxxx",
  "claims": [
    {"text": "...", "claim_type": "comparative", "testable": true,
     "expected_verdict": "consistent_at_reduced_scale"}
  ]
}
```

Métricas:
- Extracción: precisión y recall de claims frente a la anotación (emparejamiento con el LLM como juez más revisión manual de una muestra).
- Clasificación: accuracy de `claim_type` y `testable`.
- Ejecución: porcentaje de claims con código que corre sin errores, intentos de debug medios.
- Veredicto: acuerdo con `expected_verdict`.
- Coste: tokens y minutos de cómputo por claim.

Objetivo: 10-15 papers anotados, incluidos algunos con replicaciones fallidas conocidas (p. ej. informes del ML Reproducibility Challenge).

## 10. Fases

Cada fase termina con tests pasando, un commit y una actualización breve de `CLAUDE.md`.

**Fase 0. Esqueleto**
- `uv init`, estructura de carpetas, `pyproject.toml`, ruff, mypy, pytest, `.env.example`, CLI vacía.
- Aceptación: `uv run claimscope --help` funciona; CI local (`ruff`, `mypy`, `pytest`) en verde.

**Fase 1. Ingesta y extracción de claims**
- Nodos `ingest` y `extract_claims`, grafo lineal mínimo.
- Aceptación: sobre 2 papers de prueba produce `claims.json` válido; tests con el LLM mockeado.

**Fase 2. Triage, plan e interrupt**
- `triage`, `design_plan`, `review` con checkpointer SQLite.
- Aceptación: se puede interrumpir, cerrar el proceso y reanudar desde la CLI con `claimscope resume <thread_id>`.

**Fase 3. Sandbox y ejecución**
- Docker runner, `codegen` en modo `from_scratch`, `execute` con dry run, bucle `debug`.
- Aceptación: un claim de juguete (p. ej. "dropout reduce la diferencia entre loss de train y test en un MLP pequeño en MNIST") corre de punta a punta en CPU en menos de 15 minutos.

**Fase 4. Análisis e informe**
- `stats.py`, `analyze`, `report`.
- Aceptación: informe Markdown completo para el claim de juguete; tests unitarios de la regla de veredicto con datos sintéticos.

**Fase 5. Modo repo oficial**
- `codegen` en modo `official_repo`: clonar, localizar configs, reducir escala.
- Aceptación: un paper real con código oficial, elegido por el usuario, llega a veredicto.

**Fase 6. Harness de evaluación**
- `eval/run_eval.py` y métricas de la sección 9.
- Aceptación: tabla de métricas sobre al menos 5 papers anotados.

**Fase 7. Pulido**
- Trazas (Langfuse o LangSmith), README con diagrama del grafo, resultados de la evaluación, limitaciones honestas y posicionamiento frente a trabajos relacionados (PaperBench, CORE-Bench).

## 11. Reglas para Claude Code (copiar a CLAUDE.md)

- Código, nombres y comentarios en inglés. Documentación de usuario en español o inglés según indique el usuario.
- No avanzar de fase sin cumplir los criterios de aceptación.
- Ante decisiones de diseño no cubiertas por este plan, preguntar antes de implementar.
- Consultar la documentación actual de LangGraph antes de usar `interrupt`, `Command` o checkpointers.
- Nunca hardcodear claves de API; usar `.env`.
- Nunca ejecutar código generado por el LLM fuera del sandbox Docker.
- Prompts en `src/claimscope/prompts/` como ficheros, no incrustados en el código.
- Todas las salidas del LLM validadas con Pydantic; si la validación falla, reintentar una vez y registrar el error.
- Tests con el LLM mockeado por defecto; los tests que llaman a la API real van marcados y se ejecutan solo bajo demanda.
- Commits pequeños con mensajes descriptivos, uno como mínimo por fase.
