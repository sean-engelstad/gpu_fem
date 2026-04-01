import os
import runpy
import sys
import argparse
from pathlib import Path


SCRIPT_MAP = {
    ("gmg", "plate"): "../adv_elem/2_plate/1_gmg.py",
    ("gmg", "cylinder"): "../adv_elem/3_cylinder/1_gmg.py",

    ("sa_amg", "plate"): "../amg/_py_demo/3_plate_agg.py",
    ("sa_amg", "cylinder"): "../amg/_py_demo/4_cyl_agg.py",

    ("masw", "plate"): "masw/_py_demo/3_plate_masw.py",
    ("masw", "cylinder"): "masw/_py_demo/4_cyl_masw.py",

    # ("feti", "plate"): "",
    # ("feti", "cylinder"): "",

    # ("bddc", "plate"): "",
    # ("bddc", "cylinder"): "",
}


def run_script(script_path: Path, thick: float) -> None:
    """
    Run a target script while passing the thickness argument.
    """
    old_cwd = os.getcwd()
    old_argv = sys.argv.copy()

    os.chdir(script_path.parent)

    # forward thickness to the child script
    sys.argv = [script_path.name, "--thick", str(thick)]

    try:
        runpy.run_path(script_path.name, run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run solver comparison case."
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=["gmg", "sa_amg", "masw", "feti", "bddc"],
        help="Solver method"
    )

    parser.add_argument(
        "--geom",
        required=True,
        choices=["plate", "cylinder"],
        help="Geometry"
    )

    parser.add_argument(
        "--thick",
        type=float,
        default=1e-1,
        help="Shell thickness (default: 1e-1)"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    method = args.method.lower()
    geom = args.geom.lower()
    thick = args.thick

    key = (method, geom)

    if key not in SCRIPT_MAP:
        raise SystemExit(
            f"No script registered yet for method={method!r}, geom={geom!r}.\n"
            "This solver may not be implemented yet."
        )

    script = (Path(__file__).resolve().parent / SCRIPT_MAP[key]).resolve()

    if not script.exists():
        raise FileNotFoundError(f"Script does not exist: {script}")

    print(f"\nRunning comparison case")
    print(f"method   : {method}")
    print(f"geometry : {geom}")
    print(f"thickness: {thick}")
    print(f"script   : {script}\n")

    run_script(script, thick)


if __name__ == "__main__":
    main()