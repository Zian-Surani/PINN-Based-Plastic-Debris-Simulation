"""Machine and source provenance without relying on a Git checkout."""

import hashlib
import os
import platform
import subprocess
from pathlib import Path


def provenance():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "config.py", root / "app.py", root / "requirements.txt"]
    for folder in (
        "core",
        "simulation",
        "hydrology",
        "visualization",
        "scripts",
        "utils",
        "tests",
    ):
        paths.extend((root / folder).glob("*.py"))
    cpu = platform.processor()
    if os.name == "nt":
        try:
            cpu = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor).Name",
                ],
                text=True,
                timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return dict(
        cpu=cpu,
        logical_processors=os.cpu_count(),
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)
        },
    )
