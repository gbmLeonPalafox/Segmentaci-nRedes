"""Versioned editorial profiles. Published revisions are immutable and saved atomically."""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import HighlightOptions

TEMPLATES = Path(__file__).parent / "prompt_templates"
MASTER_RELEASE = "gbm-selection-v1"


def render_brief(template):
    """Bind template slots to structured inputs without interpolating untrusted source text."""
    return template.replace("{{TRANSCRIPCION_WHISPER}}", "La transcripción está en el campo units de la entrada.").replace(
        "{{ACCION_OBJETIVO_O_CONTEXTO_SI_EXISTE}}",
        "El contexto opcional se entrega en campaign_context. Si está vacío, no hay acción de campaña definida.",
    )


class PromptRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_version: int = Field(ge=1)
    objective: str = Field(min_length=10, max_length=12000)
    min_seconds: int = Field(ge=1, le=300)
    max_seconds: int = Field(ge=5, le=600)

    @model_validator(mode="after")
    def range_valid(self):
        if self.min_seconds > self.max_seconds:
            raise ValueError("Minimum duration must not exceed maximum duration.")
        return self


def initial_profiles():
    definitions = [
        (
            "acquisition-awareness",
            "acquisition",
            "Atraer atención de nuevos clientes",
            15,
            45,
            (
                "Objetivo: atraer la atención de posibles nuevos clientes que aún no conocen la propuesta. "
                "Selecciona clips con un gancho claro desde el inicio: un problema reconocible, una pregunta "
                "relevante, una historia breve o un hallazgo sorprendente respaldado por el video. Prioriza "
                "lenguaje sencillo, valor inmediato y curiosidad por conocer más. Cada clip debe entenderse "
                "sin contexto previo y cerrar una idea completa. Evita introducciones largas, jerga y "
                "sensacionalismo. No inventes beneficios, afirmaciones ni llamadas a la acción ausentes del video."
            ),
        ),
        (
            "acquisition-consideration",
            "acquisition",
            "Convencer a nuevos clientes",
            20,
            60,
            (
                "Objetivo: convertir el interés de posibles nuevos clientes en intención de conocer o probar "
                "la propuesta. Selecciona clips que expliquen un beneficio concreto, demuestren cómo se "
                "resuelve un problema o respondan una objeción frecuente. Prioriza ejemplos comprensibles, "
                "evidencia presente en el video y diferencias útiles para decidir. El clip debe ser autónomo "
                "y conservar las condiciones de cualquier afirmación. Favorece un siguiente paso claro cuando "
                "esté expresado en la fuente. No inventes testimonios, resultados, urgencia ni promesas."
            ),
        ),
        (
            "activation-first-step",
            "activation",
            "Motivar la interacción de clientes actuales",
            30,
            90,
            (
                "Objetivo: motivar a clientes existentes a interactuar más con la marca, producto o servicio. "
                "Selecciona clips que conecten con sus necesidades actuales, recuerden un beneficio útil y "
                "reduzcan una barrera para participar. Prioriza preguntas relevantes, invitaciones a comentar, "
                "compartir experiencias, explorar una función o realizar una acción sencilla cuando aparezcan "
                "en el video. Cada clip debe mostrar por qué vale la pena participar y entenderse por sí "
                "mismo. Evita discursos de captación para desconocidos y no inventes incentivos ni invitaciones."
            ),
        ),
        (
            "activation-adoption",
            "activation",
            "Impulsar el uso y la participación recurrente",
            30,
            90,
            (
                "Objetivo: convencer a clientes existentes de usar más la propuesta y volver a participar. "
                "Selecciona demostraciones, consejos aplicables, funciones poco aprovechadas o soluciones a "
                "obstáculos habituales que aporten un resultado concreto. Prioriza una acción realizable, "
                "explicación completa y un beneficio relevante para quien ya es cliente. Favorece contenido "
                "que invite a probar, retomar o profundizar el uso cuando el video lo permita. No asumas "
                "funciones, resultados, recompensas ni llamadas a la acción que no existan en la fuente."
            ),
        ),
    ]
    profiles = [
        {
            "id": key,
            "area": area,
            "name": name,
            "status": "generic",
            "revisions": [
                {
                    "version": 1,
                    "objective": objective,
                    "min_seconds": minimum,
                    "max_seconds": maximum,
                    "created_at": None,
                }
            ],
        }
        for key, area, name, minimum, maximum, objective in definitions
    ]
    for area, name, ids, minimum, maximum in (
        ("acquisition", "Adquisición", (0, 1), 15, 60),
        ("activation", "Activación", (2, 3), 30, 90),
    ):
        profiles.append(
            {
                "id": area,
                "area": area,
                "name": name,
                "status": "master",
                "revisions": [
                    {
                        "version": 1,
                        "min_seconds": minimum,
                        "max_seconds": maximum,
                        "created_at": None,
                        "objective": " ".join(profiles[i]["revisions"][0]["objective"] for i in ids),
                    }
                ],
            }
        )
        profiles[-1]["revisions"].append({
            "version": 2, "min_seconds": minimum, "max_seconds": maximum,
            "created_at": None, "release": MASTER_RELEASE,
            "objective": (TEMPLATES / f"{area}.md").read_text(encoding="utf-8").strip(),
        })
    return profiles


class PromptManager:
    def __init__(self, root):
        self.path = root / "prompts.json"
        self.lock = threading.RLock()

    def catalog(self):
        with self.lock:
            profiles = (
                json.loads(self.path.read_text(encoding="utf-8"))
                if self.path.exists()
                else initial_profiles()
            )
            existing = {p["id"] for p in profiles}
            profiles += [p for p in initial_profiles() if p["id"] not in existing]
            for profile in profiles:
                if profile["id"] in {"acquisition", "activation"} and not any(
                    r.get("release") == MASTER_RELEASE for r in profile["revisions"]
                ):
                    profile["revisions"].append({
                        "version": profile["revisions"][-1]["version"] + 1,
                        "min_seconds": profile["revisions"][-1]["min_seconds"],
                        "max_seconds": profile["revisions"][-1]["max_seconds"],
                        "created_at": datetime.now(UTC).isoformat(), "release": MASTER_RELEASE,
                        "objective": (TEMPLATES / f"{profile['id']}.md").read_text(encoding="utf-8").strip(),
                    })
            return profiles

    def install_masters(self):
        with self.lock:
            profiles = self.catalog()
            if not self.path.exists() or json.loads(self.path.read_text(encoding="utf-8")) != profiles:
                temporary = self.path.with_suffix(".tmp")
                temporary.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(self.path)

    def publish(self, prompt_id, body):
        with self.lock:
            profiles = self.catalog()
            profile = next((p for p in profiles if p["id"] == prompt_id), None)
            if profile is None:
                raise KeyError(prompt_id)
            if profile["revisions"][-1]["version"] != body.base_version:
                raise ValueError("El prompt cambió. Recarga el catálogo antes de guardar otra versión.")
            profile["revisions"].append(
                {
                    **body.model_dump(exclude={"base_version"}),
                    "version": body.base_version + 1,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.path)
            return profile

    def resolve(self, defaults, overrides):
        values = defaults.model_dump() | overrides
        snapshot = None
        if values.get("prompt_id"):
            profile = next((p for p in self.catalog() if p["id"] == values["prompt_id"]), None)
            if profile is None:
                raise ValueError("Unknown prompt profile.")
            version = values.get("prompt_version")
            revision = (
                next((r for r in profile["revisions"] if r["version"] == version), None)
                if version
                else profile["revisions"][-1]
            )
            if revision is None:
                raise ValueError("Unknown prompt revision.")
            snapshot = {k: v for k, v in profile.items() if k != "revisions"} | revision
            values.update(
                {
                    key: revision[key]
                    for key in ("objective", "min_seconds", "max_seconds")
                    if profile["status"] != "master" or key == "objective" or key not in overrides
                }
            )
            values["prompt_version"] = revision["version"]
        elif values.get("prompt_version") is not None:
            raise ValueError("A prompt version requires a profile.")
        return HighlightOptions.model_validate(values).check_range(), snapshot
