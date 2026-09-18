# ClaimScope

Agente construido con LangGraph que, dado un paper de ML de arXiv, extrae sus afirmaciones, decide
cuáles son verificables con un presupuesto de cómputo limitado, diseña una versión reducida del
experimento, la ejecuta en un sandbox Docker y emite un veredicto estadísticamente fundamentado.

ClaimScope no reproduce papers a escala completa. Responde a una pregunta más estrecha:
**¿se sostiene esta afirmación a escala reducida?** Un resultado negativo a escala reducida no
refuta el paper, y el informe siempre lo indica.

## Estado

Pipeline completo implementado: ingesta, extracción de afirmaciones, triage, diseño del plan
reducido, aprobación humana, generación de código, ejecución en sandbox, análisis estadístico e
informe.

Cuando el paper publica código oficial, ClaimScope lo clona y genera un adaptador que lo ejecuta a
escala reducida, en lugar de reimplementar el método.

Las fases que ejecutan experimentos (3 y 5) están implementadas pero **sin verificar de punta a
punta**, porque la máquina de desarrollo no tiene Docker instalado. Requiere Docker Desktop para
ejecutar experimentos de verdad.

El plan de implementación está en [PLAN.md](PLAN.md) y las reglas de trabajo en
[CLAUDE.md](CLAUDE.md).

## Instalación

Requiere Python 3.11+ y [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
cp .env.example .env    # y rellena la clave del proveedor que uses
```

ClaimScope funciona con Anthropic (por defecto) o con Google Gemini. Elige con
`CLAIMSCOPE_PROVIDER=anthropic|google` en `.env` y rellena `ANTHROPIC_API_KEY` o `GEMINI_API_KEY`
según corresponda.

## Uso

```bash
uv run python -m claimscope.cli --help
uv run python -m claimscope.cli version
uv run python -m claimscope.cli config

# Analizar un paper: extraer afirmaciones, triarlas y diseñar el experimento.
uv run python -m claimscope.cli analyze 1706.03762 --verbose

# Reanudar una ejecución que quedó esperando aprobación.
uv run python -m claimscope.cli threads
uv run python -m claimscope.cli resume 1706.03762-ce8eced0
```

`analyze` deja el PDF, el texto, `claims.json`, `triage.json`, `plans.json` y `report.md` en
`runs/<paper_id>/`, y reutiliza esa caché en ejecuciones posteriores.

Antes de ejecutar nada, el agente se detiene y te muestra cada plan de reducción para que lo
apruebes o lo rechaces con comentarios; un rechazo vuelve al diseño con tu feedback. Si cierras el
proceso en ese punto, el estado queda en `runs/checkpoints.sqlite` y `resume <thread_id>` continúa
donde lo dejaste.

En máquinas con Smart App Control activo, los shims `.exe` del entorno virtual están bloqueados;
por eso los comandos se invocan como `python -m`. Ver [CLAUDE.md](CLAUDE.md) para el detalle.

## Desarrollo

```bash
uv run python -m ruff check .
uv run python -m mypy src
uv run python -m pytest

# Pruebas de contención del sandbox. Necesitan Docker; sin él se saltan solas.
uv run python -m pytest -m docker
```

## Cómo leer un veredicto

Por cada afirmación se ejecutan ambos brazos con el mismo presupuesto y las mismas semillas. El
efecto es la diferencia de medias, con el signo orientado en la dirección que predice el paper, y el
intervalo de confianza se calcula por bootstrap sobre las semillas.

| Veredicto | Significado |
|---|---|
| `consistent_at_reduced_scale` | El IC queda entero en la dirección predicha. |
| `not_consistent_at_reduced_scale` | El IC queda entero en la dirección contraria. |
| `inconclusive` | El IC cruza el cero, o el experimento no pudo completarse. |
| `not_testable` | Triage la descartó: afirmación absoluta, dependiente de escala o fuera de presupuesto. |

**Un resultado negativo a escala reducida no refuta el paper.** Un efecto que desaparece al
encoger el modelo, los datos o el cómputo puede seguir siendo real a escala completa; eso es
precisamente lo que este método no puede distinguir. Con 3 semillas la potencia estadística es
baja, así que `inconclusive` casi siempre significa "no hay evidencia suficiente", no "no hay
efecto". Cada informe lo dice explícitamente.

## Evaluación del agente

`eval/` mide la calidad del propio agente contra papers anotados a mano en `eval/annotations/`.

```bash
uv run python -m eval.run_eval --output runs/eval.json
uv run python -m eval.run_eval --paper 1512.03385
```

Mide precisión y recall de extracción, accuracy de clasificación, coste en llamadas y tiempo. Las
métricas de ejecución y de veredicto requieren Docker; sin él la tabla muestra `—` en lugar de
`0.00`, porque un cero afirmaría que el agente falló cuando lo cierto es que no se midió.

## Seguridad

El código que genera el modelo **nunca se ejecuta fuera del sandbox**. Cada experimento corre en un
contenedor Docker sin red, con usuario no root, sistema de ficheros raíz de solo lectura, todas las
capabilities eliminadas y límites de CPU, memoria, procesos y tiempo. Lo único que se monta es el
directorio de trabajo de ese claim.
