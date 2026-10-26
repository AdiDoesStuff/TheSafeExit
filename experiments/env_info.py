import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def write_environment_info(out_path: str = "experiments/out/environment.json") -> dict:
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": None,
        "scipy_version": None,
        "matplotlib_version": None,
        "cpu_count": os.cpu_count(),
        "ram_bytes": None,
        "git_commit": None,
        "git_dirty": None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import numpy
        info["numpy_version"] = numpy.__version__
    except Exception:
        pass
    try:
        import scipy
        info["scipy_version"] = scipy.__version__
    except Exception:
        pass
    try:
        import matplotlib
        info["matplotlib_version"] = matplotlib.__version__
    except Exception:
        pass
    try:
        import psutil
        info["ram_bytes"] = psutil.virtual_memory().total
    except Exception:
        pass
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        info["git_dirty"] = bool(status)
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    return info
