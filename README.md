# ClaimScope

Agente construido con LangGraph que, dado un paper de ML de arXiv, extrae sus afirmaciones, decide
cuáles son verificables con un presupuesto de cómputo limitado, diseña una versión reducida del
experimento, la ejecuta en un sandbox Docker y emite un veredicto estadísticamente fundamentado.

ClaimScope no reproduce papers a escala completa. Responde a una pregunta más estrecha:
**¿se sostiene esta afirmación a escala reducida?** Un resultado negativo a escala reducida no
refuta el paper, y el informe siempre lo indica.

## Estado

Fases 0 a 2 completadas: ingesta, extracción de afirmaciones, triage, diseño del plan reducido y
aprobación humana con posibilidad de reanudar.

Fase 3 (sandbox Docker, generación de código y ejecución) está implementada pero **sin verificar de
punta a punta**, porque la máquina de desarrollo no tiene Docker instalado. Requiere Docker Desktop
para ejecutar experimentos.

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

`analyze` deja el PDF, el texto, `claims.json`, `triage.json` y `plans.json` en `runs/<paper_id>/`,
y reutiliza esa caché en ejecuciones posteriores.

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

## Seguridad

El código que genera el modelo **nunca se ejecuta fuera del sandbox**. Cada experimento corre en un
contenedor Docker sin red, con usuario no root, sistema de ficheros raíz de solo lectura, todas las
capabilities eliminadas y límites de CPU, memoria, procesos y tiempo. Lo único que se monta es el
directorio de trabajo de ese claim.
