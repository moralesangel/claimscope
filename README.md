# ClaimScope

ClaimScope es un agente que toma un paper de machine learning de arXiv y responde a una pregunta
estrecha pero útil: **¿se sostiene esta afirmación cuando el experimento se encoge?**

No reproduce papers. Extrae las afirmaciones del paper, descarta las que no tienen sentido probar a
escala reducida, diseña una versión pequeña del experimento, la ejecuta en un sandbox aislado y
emite un veredicto con su intervalo de confianza.

**Un resultado negativo a escala reducida no refuta el paper.** Un efecto que desaparece al encoger
el modelo, los datos o el cómputo puede seguir siendo real a escala completa; eso es exactamente lo
que este método no puede distinguir. Cada informe lo dice explícitamente.

## Cómo funciona

```
  ingest ──► extract_claims ──► triage ──► design_plan ──► review (interrupt)
    │              │              │             ▲             │
    │              │              │             └── rechazo ──┤
    │              │              │                           │ aprobado
  arXiv +      afirmaciones   nada verificable                ▼
  PyMuPDF      estructuradas        │                      codegen
                                    │                         │
                                    │        ┌── fallo ──► execute (sandbox)
                                    │        │                │
                                    │      debug ◄────────────┤
                                    │    (máx. N)             │ ok
                                    │                         ▼
                                    └──────────────────►  analyze ──► report
                                                          bootstrap    report.md
                                                           + Welch
```

1. **ingest** descarga el PDF de arXiv, extrae el texto y detecta el repositorio oficial si existe.
2. **extract_claims** convierte el texto en afirmaciones estructuradas, cada una con su tipo, sus
   brazos, su métrica y la dirección que el paper predice.
3. **triage** decide cuáles merece la pena probar. Las afirmaciones absolutas nunca lo son; las que
   dependen de la escala tampoco.
4. **design_plan** diseña el experimento reducido, documentando cada cambio respecto al paper y por
   qué no debería invertir el efecto.
5. **review** se detiene y te enseña el plan. Apruebas, o lo rechazas con comentarios que vuelven al
   diseñador. Aquí puedes cerrar el proceso y continuar después.
6. **codegen** escribe el experimento: adapta el repositorio oficial si lo hay, o lo escribe desde
   cero si no.
7. **execute** mide primero con una ejecución corta, extrapola, y si no cabe en el presupuesto
   devuelve la afirmación al diseño en vez de empezar algo que no terminará.
8. **debug** arregla el código que falla, con un tope de intentos.
9. **analyze** compara los brazos y **report** escribe el informe.

## Instalación

Requiere Python 3.11+, [uv](https://docs.astral.sh/uv/) y Docker para ejecutar experimentos.

```bash
uv sync --all-extras
cp .env.example .env
```

ClaimScope funciona con Anthropic o con Google Gemini. En `.env`, pon
`CLAIMSCOPE_PROVIDER=anthropic|google` y rellena `ANTHROPIC_API_KEY` o `GEMINI_API_KEY`.

## Uso

```bash
# Analizar un paper.
uv run python -m claimscope.cli analyze 1706.03762

# Reanudar una ejecución que quedó esperando tu aprobación.
uv run python -m claimscope.cli threads
uv run python -m claimscope.cli resume 1706.03762-ce8eced0

# Ver la configuración resuelta, con las claves ocultas.
uv run python -m claimscope.cli config
```

Todo queda en `runs/<paper_id>/`: el PDF, el texto, `claims.json`, `triage.json`, `plans.json`, los
workspaces de cada experimento y `report.md`. La caché se reutiliza entre ejecuciones.

En máquinas con Smart App Control, los `.exe` del entorno virtual están bloqueados; por eso los
comandos se invocan como `python -m`.

## Cómo leer un veredicto

Ambos brazos se ejecutan con el mismo presupuesto, el mismo dataset y las mismas semillas. El efecto
es la diferencia de medias, con el signo orientado en la dirección que predice el paper, y el
intervalo de confianza sale de un bootstrap sobre las semillas. El test de Welch se reporta como
apoyo, no como criterio.

| Veredicto | Qué significa |
|---|---|
| `consistent_at_reduced_scale` | El intervalo queda entero en la dirección predicha. |
| `not_consistent_at_reduced_scale` | El intervalo queda entero en la dirección contraria. |
| `inconclusive` | El intervalo cruza el cero, o el experimento no pudo completarse. |
| `not_testable` | Triage la descartó, con el motivo registrado. |

Con tres semillas la potencia estadística es baja: `inconclusive` casi siempre significa "no hay
evidencia suficiente", no "no hay efecto".

## Seguridad

El código que genera el modelo **nunca se ejecuta fuera del sandbox**. Cada experimento corre en un
contenedor Docker sin red, con usuario no root, raíz de solo lectura, todas las capabilities
eliminadas, `no-new-privileges` y topes de CPU, memoria, procesos y tiempo. Lo único montado es el
directorio de trabajo de esa afirmación. Las dependencias se instalan al construir la imagen, que es
el único paso con acceso a red.

## Evaluación del agente

`eval/` mide la calidad del propio agente contra 13 papers anotados a mano en `eval/annotations/`
(90 afirmaciones).

```bash
uv run python -m eval.run_eval --output runs/eval.json
uv run python -m eval.run_eval --paper 1512.03385
```

Mide precisión y recall de extracción, accuracy de clasificación, acuerdo de veredictos y coste. El
corpus incluye casos difíciles a propósito: RoBERTa, donde **ninguna** afirmación es verificable
porque todas dependen de la escala del preentrenamiento; Vision Transformer, cuyo hallazgo central
solo existe a gran escala; y Lottery Ticket y MAML, cuyas replicaciones posteriores fueron
problemáticas.

Los tests de `tests/test_eval_metrics.py` vigilan que el corpus no se desequilibre: los cuatro tipos
de afirmación presentes, los tres veredictos representados, y al menos un paper íntegramente no
verificable. Sin afirmaciones que se espera contradecir, el harness daría nota perfecta a un agente
que siempre dice que sí.

## Trazas

Opcionales y apagadas por defecto. `CLAIMSCOPE_TRACING_ENABLED=true` con
`CLAIMSCOPE_TRACING_BACKEND=langsmith` (incluido) o `langfuse` (`uv sync --extra tracing`,
autoalojable). Si el backend está mal configurado, ClaimScope avisa y sigue sin trazas: nunca tumba
una ejecución.

## Limitaciones

Conviene ser explícito sobre lo que esto no puede hacer.

- **No es una reproducción.** Un veredicto positivo dice que la dirección del efecto sobrevivió al
  encogimiento, no que se reprodujeran los números del paper.
- **Un negativo no refuta nada.** Es la limitación central, y es irreducible.
- **Poca potencia estadística.** Tres semillas por brazo detectan efectos grandes y poco más.
- **Las afirmaciones que dependen de la escala están fuera de alcance** por construcción:
  preentrenamiento de modelos de lenguaje, capacidades emergentes, leyes de escalado.
- **El experimento reducido lo escribe un modelo** a partir de la descripción del paper, así que
  puede diferir de la implementación de los autores en formas que importan.
- **Verifica afirmaciones, no papers.** Un paper con cinco afirmaciones puede tener dos que se
  sostienen a escala reducida y tres que no son comprobables.

## Frente a trabajos relacionados

**PaperBench** (OpenAI, 2025) mide si un agente puede reproducir un paper entero de ICML desde cero,
con rúbricas escritas por los propios autores. Es una pregunta mucho más ambiciosa y mucho más cara:
cada intento requiere replicar todo el trabajo. La evaluación es de fidelidad a la reproducción.

**CORE-Bench** (Siegel et al., 2024) mide reproducibilidad computacional: dado el código y los datos
de los autores, ¿consigue el agente ejecutar el repositorio y obtener los resultados publicados? El
foco está en el entorno y las dependencias, no en si el hallazgo científico es sólido.

**ClaimScope pregunta otra cosa.** No "¿puedes reproducir esto?" sino "¿esta afirmación concreta
sobrevive a que encojamos el experimento?". Las diferencias que importan:

- La unidad es la **afirmación**, no el paper. Triarlas es parte del trabajo, y decidir que algo no
  es verificable es una respuesta legítima, no un fallo.
- El presupuesto es **deliberadamente pequeño** — minutos de CPU, no GPU-días. Eso descarta clases
  enteras de afirmaciones, y el agente debe reconocerlo en vez de intentarlo igualmente.
- La salida es **estadística y calibrada**: un efecto con intervalo de confianza, y un
  `inconclusive` honesto cuando no hay evidencia suficiente.
- No requiere el código de los autores. Lo usa si existe, y si no escribe el experimento desde la
  descripción del paper.

Ninguno sustituye al otro. PaperBench y CORE-Bench miden si la maquinaria de un paper puede volver a
funcionar; ClaimScope pregunta si una afirmación concreta se mantiene cuando se la somete a un
experimento más barato.

## Estado del proyecto

El pipeline está completo. Dos cosas **no** se han verificado nunca de extremo a extremo, y conviene
saberlo antes de confiar en un resultado:

- **La ejecución real de experimentos** está implementada y testeada contra un runner falso, pero
  nunca se ha ejecutado contra Docker, porque la máquina de desarrollo no lo tiene instalado.
  `tests/test_sandbox_integration.py` contiene las pruebas de contención reales y se saltan solas
  hasta que haya un demonio disponible.
- **La calidad del triage y de los planes con un LLM real** solo se ha observado en la extracción de
  afirmaciones. El resto está validado con modelos simulados.

Con Docker disponible: `uv run python -m pytest -m docker` y luego un `analyze` real.

El plan de implementación está en [PLAN.md](PLAN.md) y las reglas de trabajo en
[CLAUDE.md](CLAUDE.md).

## Desarrollo

```bash
uv run python -m ruff check .
uv run python -m mypy src eval
uv run python -m pytest
uv run python -m pytest -m docker   # necesita Docker; sin él se salta
```
