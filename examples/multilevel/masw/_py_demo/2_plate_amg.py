import runpy
import os
from pathlib import Path

script = Path(__file__).resolve().parent / "../../../amg/_py_demo/3_plate_agg.py"

old_cwd = os.getcwd()
os.chdir(script.parent)

try:
    runpy.run_path(script.name, run_name="__main__")
finally:
    os.chdir(old_cwd)