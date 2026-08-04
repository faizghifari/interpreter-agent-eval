"""
Quick smoke test: run 2 samples each from MAPS and OpenSubs enriched data
to verify the full pipeline works end-to-end.
"""
import os
import sys
import io

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

# Import run_eval helpers directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_eval",
    os.path.join(ROOT, "scripts", "run_eval.py")
)
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)

DATA_DIR = os.path.join(ROOT, "data", "enriched")

files_to_test = [
    ("MAPS (id→kor)", os.path.join(DATA_DIR, "id_kor_maps.jsonl")),
    ("OpenSubs (id→kor)", os.path.join(DATA_DIR, "id_kor_maps_from_opensubs.jsonl")),
    ("OpenSubs (kor→id)", os.path.join(DATA_DIR, "kor_id_maps_from_opensubs.jsonl")),
]

glotlid_model = run_eval.load_glotlid_model()

for label, path in files_to_test:
    print(f"\n{'#'*60}")
    print(f"# {label}")
    print(f"# {os.path.basename(path)}")
    print(f"{'#'*60}")
    if os.path.exists(path):
        run_eval.run_simulation_sample(path, num_samples=2, glotlid_model=glotlid_model)
    else:
        print(f"File not found: {path}")
