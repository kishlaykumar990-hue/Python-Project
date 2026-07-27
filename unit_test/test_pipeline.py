"""Unit tests for the core pieces of the pipeline: ideal-function selection,
threshold labeling, final-table mapping, the custom exception, and the
loaders. Uses small fabricated data (not the real assignment CSVs in
data/), written directly in this file, or in a temporary file created and
deleted automatically during the test itself.
"""

import math
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Make sure "src" can be imported regardless of which folder pytest is run
# from (project root, or from inside unit_test/ itself).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deviation_calculator import DeviationCalculator
from src.test_handler import DeviationRegressionHandler
from src.train_loader import TrainLoader
from src.ideal_loader import IdealLoader
from src.exceptions import MappingErrorHandling


def test_select_ideal_functions_returns_best_match():
    """Selector should pick the exact-match function over a clearly wrong one."""
    # Dummy data: y1 = [0,1,2,3] is the fake training column. Y1 = [0,1,2,3]
    # is identical to it (a deliberate perfect match), Y2 = [10,10,10,10] is
    # deliberately way off. Check the code picks Y1, not Y2.
    train_df = pd.DataFrame({"x": [0, 1, 2, 3], "y1": [0, 1, 2, 3]})
    ideal_df = pd.DataFrame({
        "X": [0, 1, 2, 3],
        "Y1": [0, 1, 2, 3],       # perfect match
        "Y2": [10, 10, 10, 10],   # bad match
    })
    result = DeviationCalculator().select_ideal_functions(train_df, ideal_df, ["y1"])
    assert result["y1"]["ideal_col"] == "Y1"
    assert result["y1"]["sse"] == 0


def test_compute_max_dev_thresholds_uses_sqrt2():
    """Threshold should be the largest training deviation times sqrt(2)."""
    # Dummy data: three fake training deviations - [1.0, 3.0, 2.0]. Check
    # the threshold comes out to the largest one (3.0) times sqrt(2).
    handler = DeviationRegressionHandler()
    train_result_df = pd.DataFrame({"dev_y1_Y1": [1.0, 3.0, 2.0]})
    thresholds = handler.compute_max_dev_thresholds(train_result_df, ["y1"], ["Y1"])
    assert thresholds["dev_Y1"] == pytest.approx(3.0 * math.sqrt(2))


def test_threshold_labeling_within_and_crosses_limit():
    """Deviations should be labeled "within limit" or "crosses limit" correctly."""
    # Dummy data: two fake test points, y = [1.0, 10.0], compared against a
    # fake ideal value of 1.0 with a threshold of 2.0. First point's
    # deviation is 0 (within), second is 9 (crosses). Check both are labeled right.
    handler = DeviationRegressionHandler()
    test_df = pd.DataFrame({
        "x": [0, 1],
        "y": [1.0, 10.0],
        "Y1_interp": [1.0, 1.0],
    })
    result = handler.prepare_output(test_df, ["Y1"], {"dev_Y1": 2.0})
    assert result.loc[0, "limit_Y1"] == "within limit"   # deviation 0 <= 2.0
    assert result.loc[1, "limit_Y1"] == "crosses limit"  # deviation 9 > 2.0


def test_create_final_table_picks_smallest_deviation():
    """When two functions qualify, the smaller deviation should win."""
    # Dummy data: one fake test point with two fake deviations - 0.5 and
    # 0.2, both "within limit". Check it picks the smaller one (0.2).
    handler = DeviationRegressionHandler()
    test_result_df = pd.DataFrame({
        "X": [0],
        "Y": [1.0],
        "dev_Y1": [0.5], "dev_Y2": [0.2],
        "limit_Y1": ["within limit"], "limit_Y2": ["within limit"],
    })
    final_df = handler.create_final_table(test_result_df, ["Y1", "Y2"])
    assert final_df.iloc[0]["No. of ideal func"] == "Y2"


def test_create_final_table_raises_mapping_error_when_nothing_maps():
    """If nothing can be mapped, the custom exception should fire."""
    # Dummy data: one fake test point deliberately given a deviation that
    # "crosses limit". Check MappingErrorHandling is raised, not ignored.
    handler = DeviationRegressionHandler()
    test_result_df = pd.DataFrame({
        "X": [0], "Y": [1.0],
        "dev_Y1": [5.0], "limit_Y1": ["crosses limit"],
    })
    with pytest.raises(MappingErrorHandling):
        handler.create_final_table(test_result_df, ["Y1"])


def test_train_loader_missing_file_returns_none():
    """Loading a missing file should return None, not crash."""
    # Dummy data: none - just a made-up file path that doesn't exist.
    with tempfile.TemporaryDirectory() as tmp_dir:
        loader = TrainLoader(engine=None)
        result = loader.load_train_csv(Path(tmp_dir) / "does_not_exist.csv")
        assert result is None


def test_ideal_loader_column_renaming():
    """Ideal CSV columns should be renamed to X, Y1, Y2... regardless of the original names."""
    # Dummy data: a tiny fake CSV, written to a temporary file just for
    # this test, with columns deliberately named something other than
    # Y1/Y2 (X, Y_Dummy_Ideal_1, Y_Dummy_Ideal_2) to prove the renaming
    # actually happens rather than the names already being correct by luck.
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "fake_ideal_data.csv"
        csv_path.write_text(
            "X,Y_Dummy_Ideal_1,Y_Dummy_Ideal_2\n"
            "0,1,2\n"
            "1,2,3\n"
            "2,3,4\n"
        )
        loader = IdealLoader(engine=None)
        df = loader.load_ideal_csv(csv_path)
        assert list(df.columns) == ["X", "Y1", "Y2"]
