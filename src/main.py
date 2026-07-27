# src/main.py

import os
import pandas as pd
from src.test_handler import DeviationRegressionHandler
from src.train_loader import TrainLoader
from src.ideal_loader import IdealLoader
from src.test_loader import TestLoader
from src.deviation_calculator import DeviationCalculator
from src.exceptions import MappingErrorHandling


def main():
    """Run the full pipeline: load the train/ideal/test CSVs, save raw data
    to the database, select the 4 best ideal functions via least squares
    (SSE), compute deviations for train and test data, apply the sqrt(2)
    threshold, build the final mapping table, and generate both Bokeh
    visualizations."""
    print("Starting project pipeline...")

    # Step 1: Initialize handlers - one per responsibility, all sharing the
    # same database engine/connection from DeviationRegressionHandler
    handler = DeviationRegressionHandler()
    train_loader = TrainLoader(handler.engine)
    ideal_loader = IdealLoader(handler.engine)
    test_loader = TestLoader()
    selector = DeviationCalculator()

    # Step 2: Load datasets - all three loaded directly from CSV, nothing
    # needs to be run beforehand
    train_df = train_loader.load_train_csv(handler.TRAIN_CSV_PATH)
    print("Training dataset loaded")

    ideal_df = ideal_loader.load_ideal_csv(handler.IDEAL_CSV_PATH)
    print("Ideal dataset loaded")

    test_df = test_loader.load_test_csv(handler.TEST_CSV_PATH)
    print("Test dataset loaded")

    if ideal_df is None or train_df is None or test_df is None:
        print("Error: Required datasets could not be loaded. Exiting.")
        return

    # Save the raw inputs to the database too
    train_loader.save_to_db(train_df, "training_data")
    ideal_loader.save_to_db(ideal_df, handler.IDEAL_TABLE)

    # Step 3: Select the 4 best ideal functions live, via least squares (SSE)
    # - this is the actual assignment requirement, not a hardcoded list
    train_columns = handler.TRAIN_COLUMNS
    selection = selector.select_ideal_functions(train_df, ideal_df, train_columns)
    ideal_columns = [selection[col]["ideal_col"] for col in train_columns]

    print("\nSelected ideal functions (via least squares / SSE):")
    for train_col in train_columns:
        info = selection[train_col]
        print(f"  {train_col} -> {info['ideal_col']} (SSE = {info['sse']:.3f}, max deviation = {info['max_dev']:.6f})")

    # Step 4: Process train.csv
    train_result_df = handler.compute_train_deviations(train_df, ideal_df, train_columns, ideal_columns)
    if train_result_df is not None:
        handler.store_results_to_db(train_result_df, handler.TRAIN_TABLE)
        handler.preview_results(train_result_df, "Train Deviations")

        # Max deviation thresholds, computed live from this training run
        # (largest deviation * sqrt(2)), instead of hardcoded
        max_dev_thresholds = handler.compute_max_dev_thresholds(train_result_df, train_columns, ideal_columns)

    # Step 5: Process test.csv
    interpolated_df = handler.interpolate_ideal_values(test_df, ideal_df, ideal_columns)
    if interpolated_df is not None:
        test_result_df = handler.prepare_output(interpolated_df, ideal_columns, max_dev_thresholds)
        handler.store_results_to_db(test_result_df, handler.TEST_TABLE)
        handler.preview_results(test_result_df, "Test Deviations")

        # Step 6: Create final mapping - custom exception raised if nothing
        # could be mapped at all
        try:
            final_df = handler.create_final_table(test_result_df, ideal_columns)
        except MappingErrorHandling as e:
            print(f"Error: {e}")
            return
        handler.store_results_to_db(final_df, "final_table")
        handler.preview_results(final_df, "Final Table")

        # Step 7: Visualization - two separate charts, both saved to outputs/
        handler.visualize_test_vs_ideal(final_df, ideal_df, ideal_columns)
        handler.visualize_train_vs_ideal(train_df, ideal_df, ideal_columns)

    print("Project pipeline finished successfully!")


if __name__ == "__main__":
    main()
