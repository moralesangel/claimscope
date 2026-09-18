# CLAUDE.md

Reglas persistentes para trabajar en ClaimScope. El plan completo está en [PLAN.md](PLAN.md);
este fichero recoge las reglas de su sección 11 más el estado actual del proyecto.

## Reglas

- Código, nombres y comentarios en inglés. Documentación de usuario en español o inglés según
  indique el usuario.
- No avanzar de fase sin cumplir los criterios de aceptación de la fase actual (PLAN.md, sección 10).
- Ante decisiones de diseño no cubiertas por el plan, preguntar antes de implementar.
- Consultar la documentación actual de LangGraph antes de usar `interrupt`, `Command` o
  checkpointers. La API cambia con frecuencia; no asumirla de memoria.
- Nunca hardcodear claves de API; usar `.env` (ver `.env.example`).
- Nunca ejecutar código generado por el LLM fuera del sandbox Docker.
- Prompts en `src/claimscope/prompts/` como ficheros `.md` versionados, no incrustados en el código.
- Todas las salidas del LLM validadas con Pydantic; si la validación falla, reintentar una vez y
  registrar el error.
- Tests con el LLM mockeado por defecto; los tests que llaman a la API real van marcados
  (`@pytest.mark.llm`) y se ejecutan solo bajo demanda.
- Commits pequeños con mensajes descriptivos, uno como mínimo por fase.

## Invariantes del dominio

Al diseñar o revisar un `ReductionPlan` (PLAN.md, sección 2):

1. Ambos brazos de una comparación reciben exactamente el mismo presupuesto, dataset, reducción y
   número de semillas.
2. Mínimo 3 semillas por brazo.
3. Se conserva la arquitectura; se reduce profundidad, anchura, datos o pasos.
4. Todo cambio respecto al paper se documenta con su justificación.

Un resultado negativo a escala reducida NO refuta el paper. El informe siempre debe indicarlo.

## Comandos

```bash
uv sync --all-extras                    # instalar dependencias
uv run python -m claimscope.cli --help  # CLI
uv run python -m ruff check .           # lint
uv run python -m ruff format .          # formato
uv run python -m mypy src               # tipos
uv run python -m pytest                 # tests (LLM mockeado)
uv run python -m pytest -m llm          # tests contra la API real, bajo demanda
```

## Entorno de esta máquina

Tres restricciones locales, con su solución ya aplicada. No hace falta volver a diagnosticarlas.

1. **`uv` no está en el PATH global.** Está en `C:\Users\angel\.local\bin`. Anteponerlo en cada
   sesión: `$env:Path = "C:\Users\angel\.local\bin;$env:Path"`.
2. **Smart App Control está en modo enforcement** (`VerifiedAndReputablePolicyState = 1`). Bloquea
   los shims `.exe` de `.venv\Scripts\` (`claimscope.exe`, `mypy.exe`, `pytest.exe`). Por eso
   **siempre se invoca con `uv run python -m <modulo>`**, nunca por el nombre del ejecutable.
   `numpy`, `scipy` y `pymupdf` sí importan (sus ruedas tienen reputación establecida).

   SAC no admite exclusiones por carpeta: se aplica con políticas de Code Integrity de kernel y
   solo tiene los estados On / Evaluation / Off. Las exclusiones de Defender son otro mecanismo
   y no sirven aquí. No perder tiempo buscando una excepción por ruta.
3. **`xxhash` está bloqueado** (su `.pyd` no está firmada; probadas las versiones 3.2.0, 3.4.1 y
   3.5.0, todas bloqueadas). No es periférico: `langgraph/types.py` y `langgraph/pregel/_algo.py`
   lo importan a nivel de módulo, así que **sin resolverlo LangGraph entero no importa**.

   **Solución aplicada:** `vendor/xxhash_pure/` implementa XXH3-128 en Python puro y
   `[tool.uv.sources]` redirige la dependencia ahí. Toda la superficie que usa el árbol de
   dependencias son dos nombres (`xxh3_128`, `xxh3_128_hexdigest`), y solo para derivar IDs de
   tarea deterministas. La implementación está validada **bit a bit contra los 12 483 vectores
   oficiales** de xxHash, así que los digests son compatibles con una instalación normal: un
   checkpoint escrito aquí se lee igual en otra máquina. `tests/test_xxhash_shim.py` protege esto
   con 287 vectores muestreados que cubren todas las rutas del algoritmo.

   Si alguna vez se trabaja en una máquina sin SAC, basta con quitar la entrada de
   `[tool.uv.sources]` y la dependencia directa de `xxhash` para volver a la rueda nativa.

   El plugin de pytest de `langsmith` sigue desactivado (`addopts = "-p no:langsmith_plugin"`)
   porque no lo usamos y las trazas son opcionales.

Otras notas:

- Hardware de desarrollo sin CUDA: las fases 1 a 5 deben funcionar en CPU con experimentos de pocos
  minutos.
- Python 3.11+ (el intérprete del sistema es 3.12).
- **LangGraph instalado: 1.2.11**, no la serie 0.2.x que sugiere el plan. La API de `interrupt`,
  `Command` y checkpointers debe consultarse para 1.x antes de la fase 2.

## Estado

- **Fase 0 (esqueleto): completada.** Estructura de paquetes, `pyproject.toml`, ruff, mypy, pytest,
  `.env.example` y CLI con `--help`.
- **Fase 1 (ingesta y extracción): completada.** Nodos `ingest` y `extract_claims`, grafo lineal,
  esquemas Pydantic, cliente LLM con validación y un reintento, y `claimscope analyze <arxiv_id>`.
- Siguiente: fase 2 (triage, plan e interrupt con checkpointer SQLite).

### Notas de la fase 1

- **`arxiv` 4.0.1 eliminó `Result.download_pdf`.** Solo expone `pdf_url`; la descarga se hace con
  `requests` en `nodes/ingest.py`.
- **Las URLs se parten a mitad en los PDFs.** PyMuPDF extrae
  `https://github.com/\ntensorflow/tensor2tensor`, así que `find_repo_url` rejunta los saltos que
  siguen a `/` o `-` antes de buscar. Solo esos: unir tras el punto final de la frase se traga la
  prosa siguiente. Ambos casos tienen test de regresión con texto real de arXiv 1706.03762.
- **`ChatAnthropic` tipa su `__init__` como `(*args, **kwargs)`**, así que mypy no valida sus
  kwargs. Se usan los alias (`model_name`, `api_key`, `timeout`, `stop`), verificados en runtime.
- **mypy apunta a Python 3.12**, no a 3.11, porque los stubs incluidos en numpy usan sintaxis de
  3.12. `requires-python` del paquete sigue siendo 3.11.
- Los tests comparten dobles en `tests/stubs.py`, con `pythonpath = ["tests"]` en el pyproject.
