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
   los shims `.exe` de `.venv\Scripts\` (`claimscope.exe`, `mypy.exe`, `pytest.exe`) y la DLL nativa
   de `xxhash`. Por eso **siempre se invoca con `uv run python -m <modulo>`**, nunca por el nombre
   del ejecutable. `numpy`, `scipy`, `pymupdf` y `langgraph` sí importan correctamente.
3. **`xxhash` está bloqueado** y lo arrastra `langsmith`, que registra un plugin de pytest que lo
   importa al arrancar. Desactivado en `pyproject.toml` con `addopts = "-p no:langsmith_plugin"`.
   No es dependencia nuestra y las trazas son opcionales, así que no afecta al proyecto.

Otras notas:

- Hardware de desarrollo sin CUDA: las fases 1 a 5 deben funcionar en CPU con experimentos de pocos
  minutos.
- Python 3.11+ (el intérprete del sistema es 3.12).
- **LangGraph instalado: 1.2.11**, no la serie 0.2.x que sugiere el plan. La API de `interrupt`,
  `Command` y checkpointers debe consultarse para 1.x antes de la fase 2.

## Estado

- **Fase 0 (esqueleto): completada.** Estructura de paquetes, `pyproject.toml`, ruff, mypy, pytest,
  `.env.example` y CLI con `--help`.
- Siguiente: fase 1 (ingesta y extracción de claims).
