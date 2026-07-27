from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10, Dark2
from bokeh.layouts import gridplot
from src.exceptions import MappingErrorHandling, safe_save_to_db, safe_load_table_from_db

class DataHandler:
    """Base class for handling the SQLite database connection and reading
    tables back from it. CSV loading for each dataset lives in its own
    dedicated loader class (TrainLoader, IdealLoader, TestLoader)."""

    def __init__(self, db_path):
        """Initialize database connection."""
        self.engine = create_engine(f"sqlite:///{db_path}")
        print(f"Connected to database: {db_path}")

    def load_table_from_db(self, table_name):
        """Load a table from the SQLite database."""
        return safe_load_table_from_db(self.engine, table_name)

class DeviationRegressionHandler(DataHandler):
    """Handles processing of test.csv against the selected ideal functions:
    interpolation, deviation calculation, the sqrt(2) threshold check,
    building the final mapping table, and visualizing the results."""

    # Class constants - resolved relative to the project folder so this runs
    # on any computer, not just the one it was originally written on
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"

    DB_PATH = OUTPUT_DIR / "function_mapping.db"
    TEST_CSV_PATH = DATA_DIR / "test.csv"
    TRAIN_CSV_PATH = DATA_DIR / "train.csv"
    IDEAL_CSV_PATH = DATA_DIR / "ideal.csv"
    IDEAL_TABLE = "ideal_training_table"
    TEST_TABLE = "test_deviations"
    TRAIN_TABLE = "train_deviations"
    TRAIN_COLUMNS = ['y1', 'y2', 'y3', 'y4']
    SQRT_2 = 1.4142135623730951  # Updated for precise Max Deviation calculations

    def __init__(self):
        """Initialize DeviationRegressionHandler with database connection."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        super().__init__(self.DB_PATH)

    def interpolate_ideal_values(self, test_df, ideal_df, ideal_columns):
        """Interpolate ideal values for test.csv."""
        # Verify that all ideal_columns exist in ideal_df
        missing_cols = [col for col in ideal_columns if col not in ideal_df.columns]
        if missing_cols:
            print(f"Error: Columns {missing_cols} not found in ideal_training_table.")
            return None

        # Initialize result DataFrame
        result_df = test_df.copy()
        for col in ideal_columns:
            result_df[f"{col}_interp"] = np.nan

        # Ensure ideal_df is sorted by X
        ideal_df = ideal_df.sort_values(by='X')

        for i, row in test_df.iterrows():
            x_test = row['x']
            y_test = row['y']
            # Debug output for x = 17.5
            if abs(x_test - 17.5) < 1e-6:
                print(f"\nDebug: For x = {x_test}, y = {y_test}, raw values from ideal_training_table:")
                try:
                    raw_row = ideal_df.loc[ideal_df['X'] == x_test]
                    if not raw_row.empty:
                        for col in ideal_columns:
                            print(f"{col}: {raw_row[col].iloc[0]}")
                    else:
                        print(f"No exact match for x = {x_test}")
                except Exception as e:
                    print(f"Error accessing raw values: {e}")

            # Check if x_test is within the range of ideal_df
            if x_test < ideal_df['X'].min() or x_test > ideal_df['X'].max():
                print(f"Warning: x={x_test} is outside ideal data range. Skipping interpolation.")
                continue

            # Find the two closest x values in ideal_df
            lower = ideal_df[ideal_df['X'] <= x_test]['X'].max()
            upper = ideal_df[ideal_df['X'] >= x_test]['X'].min()

            if lower == upper:  # Exact match
                for col in ideal_columns:
                    raw_value = ideal_df.loc[ideal_df['X'] == x_test, col].iloc[0]
                    result_df.at[i, f"{col}_interp"] = raw_value
            else:
                # Linear interpolation
                lower_row = ideal_df[ideal_df['X'] == lower].iloc[0]
                upper_row = ideal_df[ideal_df['X'] == upper].iloc[0]
                x0, x1 = lower_row['X'], upper_row['X']
                for col in ideal_columns:
                    y0, y1 = lower_row[col], upper_row[col]
                    y_interp = y0 + (y1 - y0) * (x_test - x0) / (x1 - x0)
                    result_df.at[i, f"{col}_interp"] = y_interp

            # Debug deviations for x = 17.5
            if abs(x_test - 17.5) < 1e-6:
                for col in ideal_columns:
                    print(f"dev_{col} = abs(y={y_test} - {col}_interp={result_df.at[i, f'{col}_interp']}) = {np.abs(y_test - result_df.at[i, f'{col}_interp'])}")

        return result_df

    def compute_train_deviations(self, train_df, ideal_df, train_cols, ideal_cols):
        """Compute deviations for train.csv against ideal functions."""
        # Verify that all required columns exist
        missing_train_cols = [col for col in train_cols if col not in train_df.columns]
        missing_ideal_cols = [col for col in ideal_cols if col not in ideal_df.columns]
        if missing_train_cols or missing_ideal_cols:
            print(f"Error: Missing columns in train.csv: {missing_train_cols}, ideal_training_table: {missing_ideal_cols}")
            return None

        # Initialize result DataFrame
        result_df = train_df.copy()
        for ideal_col in ideal_cols:
            result_df[ideal_col] = np.nan
        deviation_cols = [f"dev_{train_col}_{ideal_col}" for train_col, ideal_col in zip(train_cols, ideal_cols)]
        for dev_col in deviation_cols:
            result_df[dev_col] = np.nan

        # Ensure ideal_df is sorted by X
        ideal_df = ideal_df.sort_values(by='X')

        for i, row in train_df.iterrows():
            x_train = row['x']
            # Debug output for x = 17.5
            if abs(x_train - 17.5) < 1e-6:
                print(f"\nDebug: For x = {x_train} in train.csv, raw values:")
                print(f"train.csv: y1={row['y1']}, y2={row['y2']}, y3={row['y3']}, y4={row['y4']}")
                try:
                    raw_row = ideal_df.loc[ideal_df['X'] == x_train]
                    if not raw_row.empty:
                        for col in ideal_cols:
                            print(f"ideal_training_table {col}: {raw_row[col].iloc[0]}")
                    else:
                        print(f"No exact match for x = {x_train}")
                except Exception as e:
                    print(f"Error accessing raw values: {e}")

            # Check if x_train is within the range of ideal_df
            if x_train < ideal_df['X'].min() or x_train > ideal_df['X'].max():
                print(f"Warning: x={x_train} is outside ideal data range. Skipping interpolation.")
                continue

            # Find the two closest x values in ideal_df
            lower = ideal_df[ideal_df['X'] <= x_train]['X'].max()
            upper = ideal_df[ideal_df['X'] >= x_train]['X'].min()

            if lower == upper:  # Exact match
                for ideal_col in ideal_cols:
                    result_df.at[i, ideal_col] = ideal_df.loc[ideal_df['X'] == x_train, ideal_col].iloc[0]
            else:
                # Linear interpolation
                lower_row = ideal_df[ideal_df['X'] == lower].iloc[0]
                upper_row = ideal_df[ideal_df['X'] == upper].iloc[0]
                x0, x1 = lower_row['X'], upper_row['X']
                for ideal_col in ideal_cols:
                    y0, y1 = lower_row[ideal_col], upper_row[ideal_col]
                    y_interp = y0 + (y1 - y0) * (x_train - x0) / (x1 - x0)
                    result_df.at[i, ideal_col] = y_interp

            # Calculate deviations
            for train_col, ideal_col, dev_col in zip(train_cols, ideal_cols, deviation_cols):
                result_df.at[i, dev_col] = np.abs(row[train_col] - result_df.at[i, ideal_col])
                # Debug deviation for x = 17.5
                if abs(x_train - 17.5) < 1e-6:
                    print(f"{dev_col} = abs({train_col}={row[train_col]} - {ideal_col}={result_df.at[i, ideal_col]}) = {result_df.at[i, dev_col]}")

        return result_df

    def compute_max_dev_thresholds(self, train_result_df, train_cols, ideal_cols):
        """Compute the sqrt(2) mapping threshold per ideal function, from the
        largest deviation observed in the training run (train_result_df,
        produced by compute_train_deviations). Pulled out into its own
        method so it can be unit-tested directly, instead of only running
        as part of the full main.py pipeline."""
        max_dev_thresholds = {}
        for train_col, ideal_col in zip(train_cols, ideal_cols):
            dev_col = f"dev_{train_col}_{ideal_col}"
            max_dev_thresholds[f"dev_{ideal_col}"] = train_result_df[dev_col].max() * self.SQRT_2
        return max_dev_thresholds

    def prepare_output(self, test_df, ideal_columns, max_dev_thresholds):
        """Prepare test_deviations with deviations and limit checks.

        max_dev_thresholds is computed live from the training run (largest
        training deviation * sqrt(2)) and passed in - not hardcoded here."""
        result_df = test_df.copy()
        output_columns = ['x', 'y'] + [f"{col}_interp" for col in ideal_columns]

        # Calculate deviations: abs(y - YXX_interp)
        for col in ideal_columns:
            result_df[f"dev_{col}"] = np.abs(result_df['y'] - result_df[f"{col}_interp"])
        output_columns += [f"dev_{col}" for col in ideal_columns]

        # Add limit columns, using the live-computed thresholds
        for col in ideal_columns:
            dev_col = f"dev_{col}"
            limit_col = f"limit_{col}"
            threshold = max_dev_thresholds[dev_col]
            result_df[limit_col] = result_df[dev_col].apply(
                lambda x: "crosses limit" if x > threshold else "within limit"
            )
            output_columns.append(limit_col)

        # Rename columns as requested
        rename_map = {'x': 'X', 'y': 'Y'}
        rename_map.update({f"{col}_interp": col for col in ideal_columns})
        result_df = result_df.rename(columns=rename_map)

        # Update output_columns to reflect renamed columns
        output_columns = ['X', 'Y'] + [col for col in ideal_columns] + [f"dev_{col}" for col in ideal_columns] + [f"limit_{col}" for col in ideal_columns]
        result_df = result_df[output_columns]
        return result_df

    def create_final_table(self, test_result_df, ideal_columns):
        """Create final table with X, Y, Delta Y (test func), and No. of ideal func."""
        final_df = test_result_df[['X', 'Y']].copy()
        final_df['Delta Y (test func)'] = np.nan
        final_df['No. of ideal func'] = ''

        for i, row in test_result_df.iterrows():
            # Collect deviations and corresponding ideal functions where limit is "within limit"
            within_limit_devs = {}
            for col in ideal_columns:
                limit_col = f"limit_{col}"
                dev_col = f"dev_{col}"
                if row[limit_col] == "within limit":
                    within_limit_devs[col] = row[dev_col]

            # If there are deviations within limit, select the smallest one
            if within_limit_devs:
                min_dev_col = min(within_limit_devs, key=within_limit_devs.get)
                final_df.at[i, 'Delta Y (test func)'] = within_limit_devs[min_dev_col]
                final_df.at[i, 'No. of ideal func'] = min_dev_col

        # Remove rows where 'Delta Y (test func)' is NaN or 'No. of ideal func' is empty
        final_df = final_df.dropna(subset=['Delta Y (test func)', 'No. of ideal func'])
        final_df = final_df[final_df['No. of ideal func'] != '']

        # Custom exception: raised if not a single test point could be
        # mapped to any of the 4 ideal functions within the threshold
        if final_df.empty:
            raise MappingErrorHandling(
                "No test points could be mapped to any ideal function within the sqrt(2) threshold."
            )

        return final_df

    def visualize_test_vs_ideal(self, final_df, ideal_df, ideal_columns):
        """
        Create a 2x2 grid of Bokeh plots (one panel per chosen ideal function),
        each showing that ideal function's line plus the test points assigned
        to it. Splitting into a grid - instead of one combined chart - avoids
        one large-scale function (e.g. Y24) squashing the other three onto a
        flat line; each panel gets its own auto-scaled axis.
        Saved to outputs/test_vs_ideal.html.
        """
        output_file(str(self.OUTPUT_DIR / "test_vs_ideal.html"))

        ideal_palette = Dark2[len(ideal_columns)]
        color_map = dict(zip(ideal_columns, ideal_palette))
        ideal_df = ideal_df.sort_values(by='X')

        panels = []
        for col in ideal_columns:
            subset = final_df[final_df['No. of ideal func'] == col].copy()
            subset['size'] = subset['Delta Y (test func)'].apply(lambda x: 5 + 20 * x)
            subset_source = ColumnDataSource(subset)
            ideal_source = ColumnDataSource(ideal_df)

            panel = figure(title=f"Ideal {col} vs Assigned Test Points",
                            x_axis_label="X", y_axis_label="Y",
                            tools="pan,box_zoom,reset,save",
                            width=480, height=380)

            panel.line(x='X', y=col, source=ideal_source, legend_label=f"Ideal {col}",
                       color=color_map[col], line_width=2, line_dash="dashed", alpha=0.7)

            scatter_plot = panel.scatter(x='X', y='Y', size='size', source=subset_source,
                                         color=color_map[col], legend_label=f"Test ({col})",
                                         alpha=0.8)

            hover = HoverTool(renderers=[scatter_plot])
            hover.tooltips = [
                ("X", "@X"),
                ("Y", "@Y"),
                ("Delta Y", "@{Delta Y (test func)}{0.0000}"),
            ]
            panel.add_tools(hover)

            panel.legend.location = "top_left"
            panel.legend.click_policy = "hide"
            panels.append(panel)

        grid = gridplot([[panels[0], panels[1]], [panels[2], panels[3]]])
        save(grid)
        print(f"Bokeh visualization saved to {self.OUTPUT_DIR / 'test_vs_ideal.html'}")

    def visualize_train_vs_ideal(self, train_df, ideal_df, ideal_columns):
        """
        Create a 2x2 grid of Bokeh plots (one panel per training function),
        each showing the raw training points plus the matched ideal function
        line, with its own auto-scaled axis - same reasoning as
        visualize_test_vs_ideal above.
        Saved to outputs/train_vs_ideal.html.
        """
        output_file(str(self.OUTPUT_DIR / "train_vs_ideal.html"))

        train_cols = [c for c in train_df.columns if c not in ('x', 'X')]
        ideal_palette = Dark2[len(ideal_columns)]
        color_map = dict(zip(ideal_columns, ideal_palette))
        ideal_df = ideal_df.sort_values(by='X')

        panels = []
        for train_col, ideal_col in zip(train_cols, ideal_columns):
            train_source = ColumnDataSource(train_df)

            # Both series are sampled at ~400 close-together x-values, so
            # plotting the ideal function as markers at every point would
            # still overlap into a solid ribbon and hide the training line -
            # thin it out to every 10th point so individual markers stay
            # visible with clear gaps, distinguishable from color alone
            ideal_sparse = ideal_df.iloc[::10].reset_index(drop=True)
            ideal_source = ColumnDataSource(ideal_sparse)

            panel = figure(title=f"Train {train_col} vs Ideal {ideal_col}",
                            x_axis_label="X", y_axis_label="Y",
                            tools="pan,box_zoom,reset,save",
                            width=480, height=380)

            # Training data as a solid line - it's densely and evenly
            # sampled, so a line is a faithful representation and it stays
            # visible underneath the sparse ideal-function markers
            panel.line(x='x', y=train_col, source=train_source,
                       legend_label=f"Train ({train_col})",
                       color=color_map[ideal_col], line_width=2.5, alpha=0.9)

            # Ideal function as sparse hollow triangle markers - visible as
            # distinct shapes (not just color) even in black-and-white print
            panel.scatter(x='X', y=ideal_col, source=ideal_source,
                          legend_label=f"Ideal {ideal_col}",
                          line_color="black", fill_alpha=0,
                          size=9, line_width=1.5, marker="triangle")

            panel.legend.location = "top_left"
            panel.legend.click_policy = "hide"
            panels.append(panel)

        grid = gridplot([[panels[0], panels[1]], [panels[2], panels[3]]])
        save(grid)
        print(f"Bokeh visualization saved to {self.OUTPUT_DIR / 'train_vs_ideal.html'}")

    def store_results_to_db(self, df, table_name):
        """Store DataFrame to SQLite database."""
        safe_save_to_db(df, self.engine, table_name, label="results")

    def preview_results(self, df, table_name):
        """Preview DataFrame and save to CSV with deviation summaries."""
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        # Format the 'X' column as strings with one decimal place
        if 'X' in df_copy.columns:
            df_copy['X'] = df_copy['X'].map(lambda x: f"{x:.1f}")

        print(f"\n--- Preview: {table_name} (All rows) ---")
        print(df_copy)
        # Save to CSV for full inspection - always into outputs/, regardless
        # of where the command is run from
        csv_path = str(self.OUTPUT_DIR / f"{table_name.lower().replace(' ', '_')}_preview.csv")
        # Calculate largest and max deviations for train_deviations
        if table_name == "Train Deviations":
            deviation_cols = [c for c in df_copy.columns if c.startswith('dev_')]
            largest_devs = {}
            max_devs = {}
            for col in deviation_cols:
                if col in df_copy.columns:
                    largest_devs[f"Largest {col}"] = df_copy[col].max()
                    max_devs[f"Max Deviation {col}"] = largest_devs[f"Largest {col}"] * self.SQRT_2
                    print(f"Largest {col}: {largest_devs[f'Largest {col}']}")
                    print(f"Max Deviation {col}: {max_devs[f'Max Deviation {col}']}")
            # Save DataFrame to CSV, preserving the formatted 'X' column
            df_copy.to_csv(csv_path, index=False)
            # Append largest and max deviations to CSV
            with open(csv_path, 'a') as f:
                f.write("\nLargest Deviations\n")
                for col, value in largest_devs.items():
                    f.write(f"{col},{value:.6f}\n")
                f.write("\nMax Deviations\n")
                for col, value in max_devs.items():
                    f.write(f"{col},{value:.6f}\n")
        else:
            # Save DataFrame to CSV, preserving the formatted 'X' column
            df_copy.to_csv(csv_path, index=False)
        print(f"Full table saved to {csv_path} for inspection.")

