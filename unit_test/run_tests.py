"""Runs every test in test_pipeline.py and prints the results in plain
English (not raw pytest jargon) to the terminal, and also saves them to
test_results/Test_Results.txt. Single command:
python3 unit_test/run_tests.py
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
TEST_FILE = THIS_DIR / "test_pipeline.py"
RESULTS_DIR = THIS_DIR / "test_results"
RESULTS_FILE = RESULTS_DIR / "Test_Results.txt"

# Matches lines like:
# unit_test/test_pipeline.py::test_select_ideal_functions_returns_best_match PASSED [ 14%]
# Also captures the "[ 14%]" progress percentage pytest reports for each test.
RESULT_LINE = re.compile(r"::(\w+)\s+(PASSED|FAILED)\s+\[\s*(\d+)%\]")


def load_test_docstrings():
    """Import test_pipeline.py and read each test function's docstring, so
    the plain-English description shown to the reader comes directly from
    the same explanation written above each test, not a separate copy."""
    spec = importlib.util.spec_from_file_location("test_pipeline", str(TEST_FILE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    descriptions = {}
    for name in dir(module):
        if not name.startswith("test_"):
            continue
        doc = getattr(module, name).__doc__ or "No description provided."
        descriptions[name] = " ".join(line.strip() for line in doc.strip().splitlines())
    return descriptions


def build_plain_english_report(raw_output, docstrings):
    """Turn pytest's raw PASSED/FAILED output into a short, numbered,
    plain-English report anyone can read - no pytest jargon, no file
    paths, no percentages."""
    lines = ["UNIT TEST RESULTS", "=" * 60, ""]
    passed_count = 0
    total_count = 0

    for match in RESULT_LINE.finditer(raw_output):
        test_name, status, _percent = match.groups()
        total_count += 1
        if status == "PASSED":
            passed_count += 1
        description = docstrings.get(test_name, test_name)
        lines.append(f"Test Case {total_count}: {description}")
        lines.append(f"   Result: {'PASSED' if status == 'PASSED' else 'FAILED'}")
        lines.append("")

    lines.append("-" * 60)
    lines.append(f"Summary: {passed_count} of {total_count} tests passed.")
    return "\n".join(lines)


def main():
    """Run pytest on test_pipeline.py, build a short plain-English report,
    print it to the terminal, and save it to test_results/Test_Results.txt."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", str(TEST_FILE)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    raw_output = result.stdout + result.stderr

    docstrings = load_test_docstrings()
    report = build_plain_english_report(raw_output, docstrings)

    print(report)

    with open(RESULTS_FILE, "w") as f:
        f.write(report)

    print(f"\nTest results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
