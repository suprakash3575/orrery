"""Shared durable JSON and text-file primitives for experiment evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def json_value(value: object) -> Any:
    """Return a detached JSON round trip through JSON-compatible values."""

    return json.loads(json.dumps(value, sort_keys=True))


def json_object_copy(value: object) -> dict[str, Any]:
    """Return a detached JSON-compatible object or reject another top level."""

    copied = json_value(value)
    if not isinstance(copied, dict):
        raise ValueError("value must become a JSON object")
    return copied


def read_json_object(path: Path, description: str) -> dict[str, Any]:
    """Read one required JSON object and describe malformed lifecycle files."""

    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"{description} is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def write_json_atomic(path: Path, value: object) -> None:
    """Durably replace a JSON file after complete sibling contents are saved."""

    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_text_atomic(path: Path, text: str) -> None:
    """Durably replace a text file through a unique same-directory temporary."""

    # The temporary is a sibling so the final rename stays on one filesystem
    # and remains atomic. A unique name also lets concurrent writers avoid
    # sharing partially written bytes.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        # Readers now switch from the complete old file to the complete new
        # file in one rename. The directory flush makes that name change survive
        # a power loss after this function reports success.
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_exclusive(path: Path, value: object) -> None:
    """Durably create new JSON evidence while refusing to replace old bytes."""

    write_text_exclusive(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_text_exclusive(path: Path, text: str) -> None:
    """Durably create one new text file and its containing directory entry."""

    # Mode x is the immutability boundary: existing evidence is never opened
    # for replacement. Flush contents before flushing the new directory entry.
    with path.open("x") as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())
    fsync_directory(path.parent)


def fsync_directory(directory: Path) -> None:
    """Persist directory entries after creating, replacing, or renaming one."""

    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
