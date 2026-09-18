# ClaimScope

Agente construido con LangGraph que, dado un paper de ML de arXiv, extrae sus afirmaciones, decide
cuáles son verificables con un presupuesto de cómputo limitado, diseña una versión reducida del
experimento, la ejecuta en un sandbox Docker y emite un veredicto estadísticamente fundamentado.

ClaimScope no reproduce papers a escala completa. Responde a una pregunta más estrecha:
**¿se sostiene esta afirmación a escala reducida?** Un resultado negativo a escala reducida no
refuta el paper, y el informe siempre lo indica.

## Estado

Fase 1 completada: ingesta desde arXiv y extracción de afirmaciones. El plan de implementación está
en [PLAN.md](PLAN.md) y las reglas de trabajo en [CLAUDE.md](CLAUDE.md).

## Instalación

Requiere Python 3.11+ y [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
cp .env.example .env    # y rellena ANTHROPIC_API_KEY
```

## Uso

```bash
uv run python -m claimscope.cli --help
uv run python -m claimscope.cli version
uv run python -m claimscope.cli config

# Descargar un paper de arXiv y extraer sus afirmaciones.
uv run python -m claimscope.cli analyze 1706.03762 --verbose
```

`analyze` deja el PDF, el texto y `claims.json` en `runs/<paper_id>/`, y reutiliza esa caché en
ejecuciones posteriores. Requiere `ANTHROPIC_API_KEY` en `.env`.

En máquinas con Smart App Control activo, los shims `.exe` del entorno virtual están bloqueados;
por eso los comandos se invocan como `python -m`. Ver [CLAUDE.md](CLAUDE.md) para el detalle.

## Desarrollo

```bash
uv run python -m ruff check .
uv run python -m mypy src
uv run python -m pytest
```
