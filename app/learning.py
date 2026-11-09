"""Publication feedback and reviewed audience adaptations in the existing SQLite database."""

import hashlib
import json
import math
import re
import unicodedata
import uuid
from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Area = Literal["acquisition", "activation"]
GOALS = {"acquisition": "attract_new_customers", "activation": "increase_customer_interaction"}


def prompt_reference(prompt):
    if not prompt:
        return None
    return {key: prompt.get(key) for key in ("id", "version", "area", "name", "objective")} | {
        "personalization_id": (prompt.get("personalization") or {}).get("id"),
    }


def normalized(value):
    return " ".join(
        re.findall(
            r"\w+",
            "".join(c for c in unicodedata.normalize("NFKD", value.lower()) if not unicodedata.combining(c)),
        )
    )


class AdaptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    area: Area
    master_version: int | None = Field(default=None, ge=1)
    audience: str = Field(min_length=3, max_length=4000)
    min_seconds: int = Field(ge=1, le=300)
    max_seconds: int = Field(ge=5, le=600)
    prior_learning: str = Field(default="", max_length=4000)
    channel: str = Field(default="", max_length=100)
    campaign_context: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def valid_range(self):
        if not self.audience.strip() or self.min_seconds > self.max_seconds:
            raise ValueError("Check audience and duration range.")
        return self


class Approval(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: str = Field(min_length=10, max_length=12000)


class ManualAdapt(AdaptInput):
    objective: str = Field(min_length=10, max_length=12000)


class AdaptedPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: str = Field(min_length=10, max_length=12000)
    changes: list[str] = Field(max_length=12)
    assumptions: list[str] = Field(max_length=12)


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selection_revision: str = Field(min_length=64, max_length=64)
    area: Area
    audience: str = Field(min_length=3, max_length=4000)
    channel: str = Field(min_length=1, max_length=100)
    published_at: date
    stage: Literal["published", "evaluated"] = "evaluated"
    evaluation: Literal["successful", "unsuccessful", "mixed", "pending"] = "pending"
    metrics: dict[str, float] = Field(default_factory=dict, max_length=20)
    feedback: str = Field(default="", max_length=4000)
    evidence_source: Literal["user_report", "measured", "synthetic"] = "user_report"

    @model_validator(mode="after")
    def valid_result(self):
        if not self.audience.strip() or not self.channel.strip():
            raise ValueError("Audience and channel are required.")
        if any(not math.isfinite(v) or v < 0 for v in self.metrics.values()):
            raise ValueError("Metrics must be finite and non-negative.")
        if any(not k.strip() or len(k) > 80 for k in self.metrics):
            raise ValueError("Invalid metric name.")
        if self.stage == "published" and self.evaluation != "pending":
            raise ValueError("Evaluate the publication before assigning a result.")
        if self.evidence_source == "measured" and not self.metrics:
            raise ValueError("Measured results require metrics.")
        return self


class LearningStore:
    def __init__(self, store):
        self.store = store
        with store.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS clip_results (
                id TEXT PRIMARY KEY, job_id TEXT NOT NULL, area TEXT NOT NULL,
                audience_key TEXT NOT NULL, goal TEXT NOT NULL, stage TEXT NOT NULL, body TEXT NOT NULL)""")
            db.execute(
                "CREATE INDEX IF NOT EXISTS result_lookup ON clip_results(area, goal, stage, audience_key)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS personalizations (id TEXT PRIMARY KEY, body TEXT NOT NULL)"
            )

    def add_result(self, record):
        record = {"id": uuid.uuid4().hex, "created_at": datetime.now(UTC).isoformat(), **record}
        with self.store.lock, self.store.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO clip_results VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record["id"],
                    record["job_id"],
                    record["area"],
                    normalized(record["audience"]),
                    record["goal"],
                    record["stage"],
                    json.dumps(record, ensure_ascii=False),
                ),
            )
        return record

    def selections(self, result):
        area = (result.get("prompt") or {}).get("area")
        if area not in GOALS:
            area = "unspecified"
        revision = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        for clip in result["highlights"]:
            self.add_result(
                {
                    "id": hashlib.sha256(f"{result['job_id']}:{revision}:{clip['id']}".encode()).hexdigest(),
                    "job_id": result["job_id"],
                    "clip_id": clip["id"],
                    "selection_revision": revision,
                    "area": area,
                    "goal": GOALS.get(area, "unspecified"),
                    "stage": "selected",
                    "evaluation": "pending",
                    "audience": result["options"].get("audience_context", ""),
                    "evidence_source": "selection",
                    "clip": clip,
                    "prompt": prompt_reference(result.get("prompt")),
                    "options": result["options"],
                }
            )

    def results(self, job_id):
        with self.store.connect() as db:
            rows = db.execute("SELECT body FROM clip_results WHERE job_id=? ORDER BY rowid DESC", (job_id,))
            return [json.loads(row[0]) for row in rows]

    def update_result(self, result_id, job_id, clip_id, changes):
        with self.store.lock, self.store.connect() as db:
            row = db.execute(
                "SELECT body FROM clip_results WHERE id=? AND job_id=?", (result_id, job_id)
            ).fetchone()
            if not row:
                raise KeyError(result_id)
            record = json.loads(row[0])
            if record["clip_id"] != clip_id or record["stage"] == "selected":
                raise KeyError(result_id)
            if changes["selection_revision"] != record["selection_revision"]:
                raise ValueError("El resultado pertenece a otra selección.")
            previous = {k: record[k] for k in changes if k in record}
            record.setdefault("history", []).append(previous)
            record.update(changes, updated_at=datetime.now(UTC).isoformat())
            record["goal"] = GOALS[record["area"]]
            db.execute(
                "UPDATE clip_results SET area=?, audience_key=?, goal=?, stage=?, body=? WHERE id=?",
                (
                    record["area"],
                    normalized(record["audience"]),
                    record["goal"],
                    record["stage"],
                    json.dumps(record, ensure_ascii=False),
                    result_id,
                ),
            )
        return record

    def evidence(self, area, audience, channel="", limit=8):
        key = normalized(audience)
        words = {w for w in key.split() if len(w) > 3} - {
            "para",
            "como",
            "with",
            "that",
            "clientes",
            "personas",
        }
        with self.store.connect() as db:
            # Indexed filtering narrows the history before decoding publication records.
            exact = db.execute(
                "SELECT body FROM clip_results WHERE area=? AND goal=? AND stage='evaluated' "
                "AND audience_key=? ORDER BY rowid DESC LIMIT 100",
                (area, GOALS[area], key),
            ).fetchall()
            other = db.execute(
                "SELECT body FROM clip_results WHERE area=? AND goal=? AND stage='evaluated' "
                "AND audience_key!=? ORDER BY rowid DESC LIMIT 200",
                (area, GOALS[area], key),
            ).fetchall()
        ranked = []
        for row, match in [(r, "same_audience") for r in exact] + [(r, "similar_audience") for r in other]:
            item = json.loads(row[0])
            if item["evaluation"] == "pending":
                continue
            tokens = {w for w in normalized(item["audience"]).split() if len(w) > 3} - {
                "para",
                "como",
                "with",
                "that",
                "clientes",
                "personas",
            }
            similarity = len(words & tokens) / max(1, len(words | tokens))
            if match == "similar_audience" and similarity < 0.25:
                continue
            # Keep references to original records, without recursively embedding past adaptations.
            item = {
                k: item.get(k)
                for k in (
                    "id",
                    "job_id",
                    "clip_id",
                    "selection_revision",
                    "area",
                    "goal",
                    "audience",
                    "channel",
                    "published_at",
                    "evaluation",
                    "metrics",
                    "feedback",
                    "evidence_source",
                )
            } | {"clip": {k: item["clip"].get(k) for k in ("title", "text", "duration")}}
            text = item["clip"].get("text") or ""
            item["clip"]["text"] = text[:4000]
            item["clip"]["text_truncated"] = len(text) > 4000
            item["relevance"] = match
            item["evidence_label"] = (
                "synthetic_hypothesis" if item["evidence_source"] == "synthetic" else "reported_result"
            )
            score = (
                match == "same_audience",
                item["evidence_source"] != "synthetic",
                bool(channel) and normalized(item["channel"]) == normalized(channel),
                item["evidence_source"] == "measured",
                similarity,
                item["published_at"],
            )
            ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        # Reserve room for both positive and negative evidence within each relevance tier.
        chosen = []
        for relevance in ("same_audience", "similar_audience"):
            pool = [item for _, item in ranked if item["relevance"] == relevance]
            for outcome in ("successful", "unsuccessful", "mixed"):
                candidate = next((i for i in pool if i["evaluation"] == outcome), None)
                if candidate and len(chosen) < limit:
                    chosen.append(candidate)
            chosen.extend(i for i in pool if i not in chosen and len(chosen) < limit)
            if len(chosen) >= limit:
                break
        return chosen

    def save_variant(self, record):
        with self.store.lock, self.store.connect() as db:
            key = normalized(record["audience"])
            row = db.execute(
                "SELECT MAX(json_extract(body, '$.version')) FROM personalizations "
                "WHERE json_extract(body, '$.area')=? AND json_extract(body, '$.audience_key')=?",
                (record["area"], key),
            ).fetchone()
            record.update(version=(row[0] or 0) + 1, audience_key=key)
            db.execute("INSERT INTO personalizations VALUES (?, ?)", (record["id"], json.dumps(record)))
        return record

    def variant(self, variant_id):
        with self.store.connect() as db:
            row = db.execute("SELECT body FROM personalizations WHERE id=?", (variant_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def variants(self):
        with self.store.connect() as db:
            return [
                json.loads(row[0])
                for row in db.execute("SELECT body FROM personalizations ORDER BY rowid DESC LIMIT 100")
            ]

    def approve(self, variant_id, objective):
        with self.store.lock, self.store.connect() as db:
            record = self.variant(variant_id)
            if record is None:
                raise KeyError(variant_id)
            if record.get("reviewed_at"):
                raise ValueError("La versión ya fue guardada. Genera otra propuesta para cambiarla.")
            record.update(objective=objective, reviewed_at=datetime.now(UTC).isoformat())
            db.execute("UPDATE personalizations SET body=? WHERE id=?", (json.dumps(record), variant_id))
        return record
