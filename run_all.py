"""
Run the full pipeline: analysis -> dashboard data export -> frontend dev server.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "app" / "frontend"


def run_all() -> None:
    # Phase 1: Analysis pipeline
    print("\n=== Running analysis pipeline ===\n")
    subprocess.run([sys.executable, ROOT / "main.py"], check=True)

    # Phase 2: Dashboard data export
    print("\n=== Exporting dashboard data ===\n")
    for script in [
        "export_dashboard_data.py",
        "whatif_simulator.py",
        "reassignment_data.py",
    ]:
        subprocess.run(
            [sys.executable, ROOT / "app" / "backend" / script], check=True
        )

    # Phase 3: Install frontend deps if needed, then launch dev server
    print("\n=== Starting dashboard dev server ===\n")
    if not (FRONTEND / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=FRONTEND, check=True)
    subprocess.run(["npm", "run", "dev"], cwd=FRONTEND)


if __name__ == "__main__":
    run_all()
