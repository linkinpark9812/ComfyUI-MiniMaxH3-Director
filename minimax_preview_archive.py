"""Durable sampling previews. No ComfyUI, GPU, or web-server imports."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import unicodedata
from uuid import uuid4
from zipfile import ZipFile, ZIP_STORED

DEFAULT_FOLDER = "MiniMaxH3-preview"
_PROCESS_ID = uuid4().hex
_RUN_ID = re.compile(r"\d{8}T\d{12}Z_[0-9a-f]{12}\Z")
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)", re.I)


def normalize_folder(value):
    """A portable single path component, including Windows device-name handling."""
    value = unicodedata.normalize("NFC", str(value or "")).strip()
    value = "".join("_" if c in '<>:"/\\|?*' or unicodedata.category(c).startswith("C")
                    else c for c in value)
    # Bound UTF-8 bytes as well as characters (filesystem component limits differ).
    value = value.encode("utf-8")[:80].decode("utf-8", errors="ignore").rstrip(" .")
    if not value or value in (".", ".."):
        return DEFAULT_FOLDER
    if _RESERVED.match(value):
        value = "_" + value.encode("utf-8")[:79].decode("utf-8", errors="ignore").rstrip(" .")
    return value


def archive_root(output_root):
    root = Path(output_root).resolve()
    return contained(root, root / "MiniMaxH3" / "previews")


def contained(root, path):
    """Reject traversal and existing symlinks/junctions that escape the root."""
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Archive path escapes its output directory")
    return path


def run_directory(output_root, folder, run_id):
    if not isinstance(folder, str) or normalize_folder(folder) != folder:
        raise ValueError("Invalid archive folder")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid archive run ID")
    root = archive_root(output_root)
    return contained(root, root / folder / run_id)


def _atomic_write(path, data):
    temporary = path.with_name(path.name + ".part")
    created = False
    try:
        # Exclusive create also refuses pre-existing links at the temporary path.
        with temporary.open("xb") as stream:
            created = True
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if created and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink(missing_ok=True)


def new_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_") + uuid4().hex[:12]


def read_manifest(output_root, folder, run_id):
    directory = run_directory(output_root, folder, run_id)
    path = contained(directory, directory / "manifest.json")
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("run_id") != run_id or manifest.get("folder_name") != folder:
        raise ValueError("Archive identity does not match its manifest")
    # A killed server cannot close its ZIP. Report that honestly after a restart.
    if manifest.get("sampling_status") == "running" and manifest.get("process_id") != _PROCESS_ID:
        manifest.update(sampling_status="interrupted", archive_state="unavailable",
                        error="Server stopped before this archive was finalized")
    manifest.pop("process_id", None)
    return manifest


def artifact_path(output_root, folder, run_id, manifest, step=None, extension="webp"):
    directory = run_directory(output_root, folder, run_id)
    if step is None:
        if manifest.get("archive_state") != "ready":
            raise RuntimeError("Archive is not ready")
        filename = folder + ".zip"
    else:
        if type(step) is not int or step < 1:
            raise ValueError("Invalid step")
        if extension not in ("webp", "wav"):
            raise ValueError("Invalid preview file type")
        record = next((r for r in manifest["artifacts"] if r["step"] == step), None)
        if record is None or (extension == "wav" and not record.get("audio")):
            raise FileNotFoundError("Step was not saved")
        filename = f"Step{step:03d}.{extension}"
    path = contained(directory, directory / filename)
    if not path.is_file():
        raise FileNotFoundError(filename)
    return path


class PreviewArchiveSession:
    """One sampler invocation; never attach mutable sessions to cached model patches."""

    def __init__(self, output_root, folder_name, node_id, context=None, settings=None):
        folder = normalize_folder(folder_name)
        parent = contained(archive_root(output_root), archive_root(output_root) / folder)
        parent.mkdir(parents=True, exist_ok=True)
        while True:
            run_id = new_run_id()
            self.directory = run_directory(output_root, folder, run_id)
            try:
                self.directory.mkdir()
                break
            except FileExistsError:
                continue
        self.manifest = {
            "schema_version": 2, "process_id": _PROCESS_ID,
            "run_id": run_id, "folder_name": folder, "requested_name": folder_name,
            "node_id": str(node_id), **(context or {}), "settings": settings or {},
            "created_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
            "sampling_status": "running", "archive_state": "recording",
            "total_steps": 0, "last_step": 0, "artifacts": [], "capture_failures": [],
            "error": None,
        }
        self._finalized = False
        self.persist()

    def persist(self):
        path = contained(self.directory, self.directory / "manifest.json")
        _atomic_write(path, json.dumps(self.manifest, ensure_ascii=False, indent=2).encode("utf-8"))

    def snapshot(self):
        # Events are queued for another thread; never enqueue our mutable manifest.
        result = json.loads(json.dumps(self.manifest))
        result.pop("process_id", None)
        return result

    def save_step(self, step, data, metadata, audio_data=None, audio_metadata=None):
        if self._finalized:
            raise RuntimeError("Archive has already been finalized")
        if type(step) is not int or step < 1 or not data:
            raise ValueError("A positive step and encoded WebP are required")
        existing = next((r for r in self.manifest["artifacts"] if r["step"] == step), None)
        if existing:
            return existing
        filename = f"Step{step:03d}.webp"
        _atomic_write(contained(self.directory, self.directory / filename), data)
        record = {**metadata, "step": step, "filename": filename, "bytes": len(data)}
        # Keep the successfully saved file in memory even if manifest persistence fails;
        # finalization can still put it in the ZIP and retry the manifest.
        self.manifest["artifacts"].append(record)
        self.manifest["artifacts"].sort(key=lambda r: r["step"])
        try:
            if audio_data is not None:
                audio_filename = f"Step{step:03d}.wav"
                _atomic_write(contained(self.directory, self.directory / audio_filename), audio_data)
                record["audio"] = {**(audio_metadata or {}), "filename": audio_filename,
                                   "bytes": len(audio_data)}
        finally:
            # A failed WAV write must not hide an already complete WebP.
            self.persist()
        return record

    def finalize(self, sampling_status, error_summary=None):
        if self._finalized:
            return self.snapshot()
        self._finalized = True
        self.manifest.update(sampling_status=sampling_status, archive_state="finalizing",
                             finished_at=datetime.now(timezone.utc).isoformat())
        if error_summary:
            self.manifest["error"] = error_summary
        name = self.manifest["folder_name"]
        archive = contained(self.directory, self.directory / (name + ".zip"))
        temporary = archive.with_name(archive.name + ".part")
        created = False
        try:
            with temporary.open("xb") as stream:
                created = True
                with ZipFile(stream, "w", compression=ZIP_STORED, allowZip64=True) as bundle:
                    if not self.manifest["artifacts"]:
                        bundle.writestr(name + "/", b"")
                    for record in self.manifest["artifacts"]:
                        for extension in ("webp", "wav") if record.get("audio") else ("webp",):
                            filename = f"Step{record['step']:03d}.{extension}"
                            path = contained(self.directory, self.directory / filename)
                            bundle.write(path, arcname=name + "/" + filename)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, archive)
            self.manifest["archive_state"] = "ready"
        except Exception as exc:
            self.manifest.update(archive_state="unavailable", error=f"ZIP creation failed: {exc}")
        finally:
            if created and temporary.is_file() and not temporary.is_symlink():
                temporary.unlink(missing_ok=True)
        self.persist()
        return self.snapshot()
