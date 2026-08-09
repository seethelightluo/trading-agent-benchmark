"""Crash-safe JSON persistence helpers used by the AlphaCrafter sandbox."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    keep_backup: bool = True,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        if keep_backup and destination.exists():
            backup = destination.with_suffix(destination.suffix + ".bak")
            backup_fd, backup_tmp_name = tempfile.mkstemp(
                prefix=f".{backup.name}.", suffix=".tmp", dir=destination.parent
            )
            os.close(backup_fd)
            backup_tmp = Path(backup_tmp_name)
            try:
                shutil.copyfile(destination, backup_tmp)
                with backup_tmp.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(backup_tmp, backup)
            finally:
                backup_tmp.unlink(missing_ok=True)

        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
    default: Any = str,
    keep_backup: bool = True,
) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent, default=default),
        keep_backup=keep_backup,
    )


def load_json(path: str | Path, *, default: Any = None, recover: bool = True) -> Any:
    """Load JSON, recovering from the adjacent .bak file when necessary."""
    source = Path(path)
    candidates = [source]
    if recover:
        candidates.append(source.with_suffix(source.suffix + ".bak"))
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            last_error = exc
            continue
        if candidate != source:
            atomic_write_json(source, payload, keep_backup=False)
        return payload
    if default is not None:
        return default
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(source)


def atomic_unlink(path: str | Path) -> None:
    target = Path(path)
    target.unlink(missing_ok=True)
    _fsync_directory(target.parent)
