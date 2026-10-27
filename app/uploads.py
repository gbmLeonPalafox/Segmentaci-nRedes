"""Bounded, retryable uploads; unfinished files never enter the processing queue."""

import json
import re
import shutil

from fastapi import HTTPException

CHUNK_BYTES = 8 * 1024**2
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg", ".mts"}


class Uploads:
    def __init__(self, store, enqueue):
        self.store = store
        self.enqueue = enqueue
        self.root = store.root / ".uploads"

    def folder(self, upload_id):
        if not re.fullmatch(r"[a-f0-9]{32}", upload_id):
            raise HTTPException(404, "Upload not found.")
        folder = self.root / upload_id
        if not folder.is_dir():
            # A completion response may have been lost after moving the file.
            folder = self.store.root / upload_id
        if not (folder / "upload.json").is_file():
            raise HTTPException(404, "Upload not found.")
        return folder

    def create(self, job, size):
        self.root.mkdir(exist_ok=True)
        folder = self.root / job["id"]
        folder.mkdir()
        (folder / "upload.json").write_text(json.dumps({"job": job, "size": size}))
        (folder / job["source_file"]).touch()
        return {"id": job["id"], "offset": 0, "chunk_bytes": CHUNK_BYTES}

    def append(self, upload_id, offset, data):
        folder = self.folder(upload_id)
        metadata = json.loads((folder / "upload.json").read_text())
        source = folder / metadata["job"]["source_file"]
        current = source.stat().st_size
        if offset < 0 or not data or len(data) > CHUNK_BYTES or offset + len(data) > metadata["size"]:
            raise HTTPException(413, "Invalid upload chunk size.")
        if offset < current:
            # Retry after a lost acknowledgement: accept only the identical bytes.
            with source.open("rb") as existing:
                existing.seek(offset)
                if existing.read(len(data)) == data:
                    return {"offset": current}
        if offset != current:
            raise HTTPException(409, "Upload offset mismatch. Start the upload again.")
        if self.store.get(upload_id):
            raise HTTPException(409, "This upload is already complete.")
        with source.open("r+b") as output:
            output.seek(current)
            try:
                output.write(data)
            except OSError:
                output.truncate(current)
                raise
        return {"offset": current + len(data)}

    def complete(self, upload_id):
        existing = self.store.get(upload_id)
        if existing:
            return existing
        folder = self.folder(upload_id)
        metadata = json.loads((folder / "upload.json").read_text())
        job = metadata["job"]
        if (folder / job["source_file"]).stat().st_size != metadata["size"]:
            raise HTTPException(409, "The video has not finished uploading.")
        destination = self.store.root / upload_id
        if folder != destination:
            folder.rename(destination)
        return self.enqueue(job)

    def cancel(self, upload_id):
        if not re.fullmatch(r"[a-f0-9]{32}", upload_id):
            raise HTTPException(404, "Upload not found.")
        folder = self.root / upload_id
        if folder.is_dir():
            shutil.rmtree(folder)
        return {"cancelled": True}
