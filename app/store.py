import json
import sqlite3
import threading
from pathlib import Path


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, body TEXT NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.root / "jobs.sqlite3", timeout=30)

    def save(self, job):
        with self.lock, self.connect() as db:
            db.execute("INSERT OR REPLACE INTO jobs VALUES (?, ?)", (job["id"], json.dumps(job)))
        return job

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT body FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def update(self, job_id, expected_status=None, expected_fields=None, **fields):
        with self.lock, self.connect() as db:
            row = db.execute("SELECT body FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            job = json.loads(row[0])
            if expected_status is not None and job["status"] != expected_status:
                return None
            if expected_fields and any(job.get(k) != v for k, v in expected_fields.items()):
                return None
            job.update(fields)
            db.execute("UPDATE jobs SET body = ? WHERE id = ?", (json.dumps(job), job_id))
        return job

    def all(self):
        with self.connect() as db:
            rows = db.execute("SELECT body FROM jobs ORDER BY rowid DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete(self, job_id):
        with self.lock, self.connect() as db:
            db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
