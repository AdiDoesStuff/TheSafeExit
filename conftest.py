import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)


def pytest_configure(config):
    if getattr(config.option, "basetemp", None) is None:
        config.option.basetemp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".pytest_tmp"
        )
