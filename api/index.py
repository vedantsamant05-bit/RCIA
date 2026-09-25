import sys
from pathlib import Path

# Ensure the root project directory is on sys.path for serverless execution
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app.main import app