"""Build the Windows desktop app with PyInstaller (one folder, no console)."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist" / "FocusTracker"


def build():
    subprocess.check_call([sys.executable, "-m", "PyInstaller", "-y", "--onedir", "--noconsole", "--name", "FocusTracker",
                           "--collect-data", "mediapipe", "--hidden-import", "focus", str(ROOT / "gui_app.py")])
    copies = ["models/face_landmarker.task", "models/yolov8n.pt", "web/models/drowsiness.json", "web/models/nudge_prior.json"]
    for rel in copies:
        src, dst = ROOT / rel, DIST / "_internal" / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            print("copied", rel)
        else:
            print("WARNING: missing", rel)
    print("Done:", DIST / "FocusTracker.exe")


if __name__ == "__main__":
    build()
