"""Shared filesystem locations derived from the installed source tree."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
