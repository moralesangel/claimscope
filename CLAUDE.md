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

## Entorno

- Python 3.11+ (mypy apunta a 3.12: los stubs de numpy usan sintaxis de esa versión).
- Desarrollo sin CUDA: los experimentos deben correr en CPU en pocos minutos.
- **LangGraph 1.x**, no la serie 0.2.x que sugiere el plan. Consultar su documentación antes de
  tocar `interrupt`, `Command` o checkpointers.
- Los comandos se invocan como `uv run python -m <modulo>`, no por el nombre del ejecutable: en
  algunas máquinas Windows los shims `.exe` del entorno virtual están bloqueados por políticas de
  integridad de código.
- `vendor/xxhash_pure/` existe porque la rueda nativa de `xxhash` puede estar bloqueada en Windows
  con Smart App Control, y LangGraph la importa a nivel de módulo. Implementa XXH3-128 en Python
  puro, validado contra los 12 483 vectores oficiales, así que los checkpoints son portables. En
  una máquina sin esa restricción, basta quitar la entrada de `[tool.uv.sources]`.

Las particularidades de una máquina concreta van en `CLAUDE.local.md`, que git ignora.

## Estado

- **Fase 0 (esqueleto): completada.** Estructura de paquetes, `pyproject.toml`, ruff, mypy, pytest,
  `.env.example` y CLI con `--help`.
- **Fase 1 (ingesta y extracción): completada.** Nodos `ingest` y `extract_claims`, grafo lineal,
  esquemas Pydantic, cliente LLM con validación y un reintento, y `claimscope analyze <arxiv_id>`.
- **Fase 2 (triage, plan e interrupt): completada.** Nodos `triage`, `design_plan` y `review` con
  `interrupt`, checkpointer SQLite, y los comandos `resume` y `threads`.
- **Fase 3 (sandbox y ejecución): implementada, ACEPTACIÓN PENDIENTE.** `Runner`,
  `DockerCPURunner`, `codegen` from_scratch, `execute` con dry run y bucle `debug`.

  **Docker no está instalado en esta máquina**, así que el criterio de aceptación (un claim de
  juguete de punta a punta en CPU en menos de 15 min) **no se ha verificado**. El código está
  completo y testeado contra un `FakeRunner`; `tests/test_sandbox_integration.py` contiene las
  pruebas de contención reales y se saltan solas mientras no haya demonio. Al instalar Docker:
  `uv run python -m pytest -m docker` y luego una ejecución real.

  Docker Desktop necesita WSL2, que a su vez requiere administrador y reiniciar. Ninguna de las
  dos cosas se puede hacer desde esta sesión.
- **Fase 4 (análisis e informe): completada.** `stats.py` (bootstrap sobre semillas, Welch de
  apoyo, regla de veredicto), nodos `analyze` y `report`. Aceptación cumplida: informe Markdown
  completo para el claim de juguete y tests de la regla de veredicto con datos sintéticos.
- **Fase 5 (modo repo oficial): implementada, ACEPTACIÓN PENDIENTE.** `repo.py` clona e inspecciona
  el repositorio del paper, y `codegen` genera un adaptador que lo ejecuta a escala reducida.
  La aceptación ("un paper real con código oficial llega a veredicto") **necesita Docker**, igual
  que la fase 3. La inspección sí está verificada contra repos reales (pytorch-cifar, nanoGPT).
- **Fase 6 (harness de evaluación): completada en lo medible.** `eval/` con anotaciones,
  emparejamiento de claims, métricas y `run_eval.py`. Tabla producida sobre 2 papers anotados.

  **Limitación conocida:** las métricas de ejecución y de veredicto no se pueden medir sin Docker.
  El harness devuelve `measured=False` y la tabla muestra `—`, no `0.00`: un cero afirmaría que el
  agente falló, cuando lo cierto es que no se midió.
- **Corpus de evaluación ampliado a 13 papers, 90 claims.** Cumple los 10-15 del plan, con tests que
  vigilan su equilibrio.
- **Fase 7 (pulido): completada.** Trazas opcionales (LangSmith y Langfuse), README reescrito con
  diagrama del grafo, limitaciones y posicionamiento frente a PaperBench y CORE-Bench.
- **Sandbox de subproceso + notebook de Colab: añadidos.** `CLAIMSCOPE_SANDBOX_BACKEND=subprocess`
  permite ejecutar donde no hay Docker. **El pipeline ya se ha ejecutado de punta a punta**: un
  experimento real corre, produce métricas y llega a veredicto (`tests/test_end_to_end.py`).
- **VALIDADO DE PUNTA A PUNTA CON UN LLM REAL.** Ejecución completa sobre arXiv 1207.0580 con
  `gemini-3.5-flash` y sandbox de subproceso: 8 claims extraídos, 7 descartados por triage con
  motivos correctos, experimento reducido a `load_digits`, 10 ejecuciones (2 brazos × 5 semillas),
  veredicto `consistent_at_reduced_scale` con efecto +0.068, IC [+0.055, +0.082], Welch p=4.4e-05.
  El informe está guardado en `docs/example-report.md`.

  El efecto medido es real, pero **el informe de esa ejecución dice "MNIST" y el script usó
  `load_digits`**. Está anotado en la cabecera de `docs/example-report.md`. Ver más abajo.
- **Integridad de los experimentos: corregida tras encontrar resultados silenciosamente falsos.**
  Una ejecución posterior produjo un veredicto de CIFAR-10 calculado sobre los dígitos de 8x8 de
  sklearn, uno de Reuters calculado con `make_classification`, y los tres scripts aleatorizaban el
  25% de las etiquetas de entrenamiento. El error de MNIST salía 61% en vez de ~5%: ambos brazos
  ajustaban etiquetas destruidas, así que ninguno podía ganar.

  **La causa era que los tres prompts se contradecían.** `design_plan` pedía preferir MNIST y
  CIFAR-10, que el sandbox no puede descargar; `codegen` prohibía sustituir; y `debug` ordenaba
  "replace that with synthetic data" ante cualquier fallo de descarga. El planificador prometía
  datos imposibles y el nodo `debug` -- que ve justo ese fallo -- los cambiaba en silencio.

  Los tres prompts ya coinciden: el planificador conoce qué datos existen sin red y debe nombrar
  el sustituto en `reduced_setup` (invariante 4), y `debug` ya no recomienda sustituir ni corromper
  etiquetas. `src/claimscope/integrity.py` revisa el script generado y anota en `errors` -- que el
  informe imprime -- toda sustitución no documentada o corrupción de etiquetas. Es orientativo, no
  bloqueante: un falso positivo que tumbara un claim costaría más que el aviso.

  **Al tocar uno de esos tres prompts, revisar los otros dos.** Cada uno es razonable por separado;
  el fallo solo aparece al componerlos, y produce una respuesta segura y equivocada en vez de un
  error. Y tras cualquier cambio, **leer el `run.py` generado**, no el informe: el informe imprime
  el plan, no lo que hizo el script.
- **Un experimento correcto puede seguir sin poder responder.** Con datos honestos, los cuatro
  claims salieron `inconclusive`: un MLP de 2 capas sobre 1797 dígitos fáciles llega al 5% de error
  sin sobreajustar, y el dropout no tiene nada que regularizar. El prompt del planificador pide
  ahora conservar el régimen donde el efecto puede aparecer (para un regularizador, una brecha
  visible entre train y test en el brazo base).
- Pendiente, y bloqueado solo por el entorno: el sandbox Docker nunca se ha ejecutado (falta WSL2),
  y el harness de evaluación no se ha corrido sobre los 13 papers.

### Notas del backend de subproceso

- **No es contención, y no hay que presentarlo como tal.** Bloquea red, limita recursos y sanea el
  entorno, pero el código que corre en ese proceso puede deshacerlo desde dentro. Es opt-in, avisa
  al preparar, y marca `unconfined_execution` para que el informe lo diga al lector.
- **El bloqueo de red se inyecta vía `sitecustomize.py`**, que Python importa al arrancar. Va en un
  directorio hermano al workspace, no dentro, para que `run.py` no lo importe por accidente.
- Se parchean `socket.connect`, `create_connection`, `getaddrinfo` y `urllib`, más las variables
  `HF_HUB_OFFLINE` y `TRANSFORMERS_OFFLINE` como segunda barrera.
- Los límites de `resource` son solo POSIX: en Windows no aplican y el test correspondiente se salta.
  En Colab (Linux) sí funcionan.
- `prepare()` instala en el entorno actual, no en una imagen. Es un efecto secundario real, y otra
  razón para que este backend sea opt-in.
- El notebook `notebooks/claimscope_colab.ipynb` verifica el bloqueo de red **antes** de gastar
  llamadas al modelo.

### Notas de la fase 7

- **Las trazas nunca pueden tumbar una ejecución.** Backend mal configurado, paquete ausente o host
  inalcanzable: se avisa y se sigue sin trazas. `tests/test_tracing.py` lo comprueba explícitamente.
- **Hay que hacer flush al salir.** Los clientes de trazas agrupan en segundo plano, y una ejecución
  corta de CLI termina antes de que se envíe nada.
- `langsmith` funciona aquí porque el shim de `xxhash` lo desbloquea; `langfuse` es un extra
  opcional (`uv sync --extra tracing`).
- **El corpus de evaluación no tenía ningún claim que se esperase contradecir.** Eso habría dado
  nota perfecta a un agente adulador. MAML aporta dos, y hay un test que impide la regresión.

### Notas de la fase 6

- **El emparejamiento es léxico por defecto, no con LLM.** El plan sugiere el LLM como juez, pero
  eso es no determinista y cuesta una llamada por par. `eval/matching.py` usa Jaccard sobre palabras
  de contenido, ponderando métrica y nombres de brazo, que es lo que distingue claims del mismo
  paper. Emparejamiento voraz y uno a uno.
- **La clasificación se puntúa solo sobre los claims emparejados.** Penalizar el tipo de un claim
  que el agente nunca extrajo contaría el mismo fallo dos veces.
- **La evaluación auto-aprueba los planes.** Un humano aprobando cada plan mediría al humano, no al
  agente. Eso implica que se ejecutaría código sin revisar, así que sin sandbox el harness **para
  antes de ejecutar** en vez de correrlo sin protección.
- `eval/` necesita `__init__.py` y está incluido en mypy (`files = ["src", "eval"]`).
- B008 de ruff está desactivado en las CLIs: `typer.Option` en los defaults es su API documentada.

### Notas de la fase 5

- **No hay un formato común de configuración.** `pytorch-cifar` solo expone `--lr` con el resto
  hardcodeado; nanoGPT usa ficheros Python en `config/`. Por eso `repo.py` no intenta *entender* el
  repo: reúne evidencia (entrypoints, flags, configs, imports) y el prompt pide al modelo que
  escriba el adaptador contra esa evidencia.
- **Los imports se infieren del AST, no del `requirements.txt`.** Muchos repos de investigación no
  declaran dependencias: `pytorch-cifar` no tiene requirements y necesita torch. Sin
  `infer_imports()` la imagen del sandbox no tendría torch y **toda ejecución fallaría en el primer
  import**. Se filtran la stdlib (vía `sys.stdlib_module_names`) y los módulos del propio repo.
- **`execute` llama a `runner.prepare()`** antes de nada. Era un hueco de la fase 3: la imagen no se
  construía nunca. Las dependencias se instalan ahí, que es el único paso con red.
- **Un repo que no se puede clonar no cuesta el claim**: se cae a `from_scratch` y se registra el
  motivo en `errors`.
- El commit del repo se guarda en `repo_commits` y aparece en el informe. Sin él, el resultado no
  es reproducible.

### Notas de la fase 4

- **El signo del efecto depende de la métrica.** `metric_direction()` detecta métricas donde menos
  es mejor (loss, error, perplexity, RMSE, FID...) por palabras completas, no por subcadenas: si no,
  "lossless" se clasificaría como loss. Equivocarse aquí invierte el veredicto en silencio.
- **El intervalo se firma antes de aplicar la regla**, de modo que positivo siempre significa "en la
  dirección que predice el paper". Así la regla de la sección 8 es una sola comparación con cero.
- **Un intervalo que toca el cero es `inconclusive`**, no consistente. Un límite exactamente en cero
  no es evidencia de dirección.
- **Todos los caminos del grafo llegan a `analyze`**, incluido "triage no seleccionó nada". Si no,
  un paper sin claims verificables no generaría informe y el silencio se leería como éxito.
- **`build_graph` aplica el serializador a cualquier checkpointer**, no solo al de SQLite. Al probar
  con `InMemorySaver` reaparecían los avisos de tipos no registrados, que en una versión futura
  serán errores.
- Los brazos deben estar emparejados: `analyze` rechaza comparar 3 semillas contra 2 y lo reporta
  como `inconclusive` con el motivo.

### Notas de la fase 3

- **Los tipos del sandbox también van en `_ALLOWED_MODULES`.** `ExecutionFailure` y
  `ExecutionResult` viajan en el estado del grafo; `ExecutionRequest` e `ImageSpec` no. El guard de
  `tests/test_session.py` detectó esta omisión al añadir `GeneratedCode` y `CodePatch`.
- **`execute` para en el primer fallo.** Un script roto falla igual para cada semilla; seguir solo
  gasta presupuesto antes de que `debug` pueda arreglarlo.
- **El dry run mide, no adivina.** Ejecuta `--steps 20`, extrapola con `full_run_steps`, y si no
  cabe en lo que queda de presupuesto marca el claim en `over_budget_claim_ids` en vez de empezar
  un estudio que no terminará.
- **El validador de código quita fences de markdown** (`strip_markdown_fences`) porque el modelo
  los añade pese a que el prompt lo prohíbe, y respeta los backticks que estén dentro del código.
- Cuidado al probar fences desde PowerShell: el backtick es su carácter de escape y corrompe la
  entrada. Usa un fichero `.py`, no `-c` con here-string.

### Notas de la fase 2

- **Al reanudar, el nodo se re-ejecuta desde el principio.** Todo lo anterior a `interrupt()` corre
  dos veces, así que esa parte no debe tener efectos secundarios y la decisión se aplica solo
  después. Está documentado en el docstring de `nodes/review.py`.
- **Los esquemas deben registrarse en `session._ALLOWED_MODULES`.** LangGraph avisa al
  deserializar tipos no registrados y los bloqueará en una versión futura, lo que dejaría las
  ejecuciones interrumpidas sin poder reanudarse. `tests/test_session.py` falla si añades un
  esquema y olvidas registrarlo; verificado con `LANGGRAPH_STRICT_MSGPACK=true`.
- **`max_retries=0` en ambos proveedores.** Sus SDKs reintentan por su cuenta (~40 s en un error de
  cuota diaria que nunca se resolverá) y ocultan los intentos de nuestro logging. `_is_transient()`
  decide mejor: distingue congestión de cuota agotada.
- **El tier gratuito de Gemini da 20 peticiones/día POR MODELO**, no por cuenta
  (`GenerateRequestsPerDayPerProjectPerModel`). Esto bloqueó el proyecto varios días hasta
  descubrirlo: con `gemini-3.6-flash` agotado, `gemini-flash-latest` respondía perfectamente y hay
  más de 20 modelos disponibles con la misma clave.

  `Settings.model_chain()` y `DEFAULT_FALLBACKS` implementan el cambio automático: si un modelo se
  queda sin cuota **o sigue congestionado tras todo el backoff**, se pasa al siguiente. Un 429 por
  minuto no cambia de modelo, porque eso sí se resuelve esperando.
- **`claimscope doctor`** comprueba clave, modelo que responde y sandbox antes de gastar cuota. Es
  lo primero que hay que ejecutar cuando algo falla.
- La regla "`absolute` nunca es verificable" se aplica en código, no se confía al modelo.

### Modelo local (Ollama)

`CLAIMSCOPE_PROVIDER=ollama` usa un modelo local: sin clave, sin cuota. Instalado y verificado con
**Qwen3 4B Q4_K_M** (2,5 GB).

**La red bloquea el CDN de Ollama.** `registry.ollama.ai` responde, pero
`r2.cloudflarestorage.com` da timeout, así que `ollama pull` falla siempre. Hugging Face sí
funciona: descarga el `.gguf` de ahí y haz `ollama create` con un Modelfile. El Modelfile debe fijar `num_ctx`.

**`num_ctx` es crítico.** Ollama usa 2048 tokens por defecto, y el prompt de extracción lleva hasta
60k caracteres (~15k tokens): el paper se truncaría en silencio. Se fija a 24.000 tanto en el
Modelfile como en `Settings.ollama_context_tokens`.

**Qué se midió (Intel Ultra 5 225H, 14 núcleos, sin GPU usable):**

| Tarea | Tiempo | Calidad |
|---|---|---|
| Extracción sobre un extracto corto | 97 s | JSON válido, pero `arms=[]` en todas |
| Triage de 3 claims | 47 s | Clasificó mal: aprobó el claim de TIMIT (dataset con licencia) |
| Extracción sobre el paper real (20k chars) | **>15 min, abortado** | — |

**Conclusión: sirve para probar el cableado, no para producción.** Tres problemas de calidad que
Gemini no tiene: no rellena `arms` (lo que rompe `codegen`), confunde `ablation` con `absolute`, y
el triage deja pasar claims no verificables. Y con un paper completo el tiempo por llamada lo hace
impracticable: una ejecución son 4+ llamadas.

Úsalo para verificar que el pipeline conecta sin gastar cuota; usa Gemini o Anthropic para juzgar
la calidad del agente.

### Proveedores de LLM

El plan fija Anthropic. Como esa cuenta no tiene saldo, `llm.py` abstrae el proveedor detrás del
`Protocol StructuredLLM` y se elige con `CLAIMSCOPE_PROVIDER` (`anthropic` | `google`), sin tocar
código. `CLAIMSCOPE_MODEL_NAME` vacío toma el modelo por defecto del proveedor.

- **La fase 1 se validó con `gemini-3.6-flash`**, no con Claude. Al volver a Anthropic hay que
  revalidar: los modelos difieren en cuántos claims extraen y en su fidelidad.
- **Los modelos `gemini-2.5-*` están retirados** para cuentas nuevas; la serie actual es 3.x.
- **`gemini-3.1-pro-preview` da 429 con `limit: 0`** en el tier gratuito: no es congestión, es que
  no está disponible. Solo `flash` funciona con esta clave.
- Las claves de Gemini pueden llevar prefijo `AQ.` además del clásico `AIza`; ambas van en la
  `x-goog-api-key`.
- `_is_transient()` distingue fallos reintentables (503, 429 por congestión) de permanentes
  (`limit: 0`, saldo agotado, auth). Reintentar un `limit: 0` no sirve de nada.

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
- **Los tests están aislados del entorno.** Una fixture `autouse` en `conftest.py` borra todas las
  variables que lee `Settings` y hace `chdir` a un directorio temporal. Sin eso, `Settings` carga
  el `.env` real del desarrollador y la suite solo pasa en su máquina. Al añadir un campo nuevo a
  `Settings`, añade su variable a `_SETTINGS_ENV_VARS`.
