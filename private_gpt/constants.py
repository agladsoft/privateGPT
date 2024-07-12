import os
from pathlib import Path

PROJECT_ROOT_PATH: Path = Path(__file__).parents[1]

FILES_DIR = os.path.join(PROJECT_ROOT_PATH, "upload_files")
os.makedirs(FILES_DIR, exist_ok=True)
os.chmod(FILES_DIR, 0o0777)
