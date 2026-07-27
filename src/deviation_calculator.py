import numpy as np
import pandas as pd


class DeviationCalculator:
    """Selects the 4 best-fit ideal functions for the training data.

    The selection criterion is least squares: for each training function,
    compare it against all 50 ideal functions and pick the one with the
    lowest sum of squared errors (SSE), as required by the assignment.
    """

    def select_ideal_functions(self, train_df, ideal_df, train_cols):
        """Compare each training function against all 50 ideal functions and
        pick the one with the lowest sum of squared errors (least squares).

        Returns a dict keyed by training column, each value holding the
        matched ideal column name, its SSE, and the largest single deviation
        observed (used later to compute the sqrt(2) mapping threshold).
        """
        merged = pd.merge(train_df, ideal_df, left_on='x', right_on='X', how='inner')
        ideal_cols = [c for c in ideal_df.columns if c != 'X']

        selection = {}
        for train_col in train_cols:
            train_values = merged[train_col].to_numpy(dtype=float)
            best_ideal_col, best_sse, best_max_dev = None, None, None
            for ideal_col in ideal_cols:
                ideal_values = merged[ideal_col].to_numpy(dtype=float)
                diff = train_values - ideal_values
                sse = float(np.sum(diff ** 2))
                if best_sse is None or sse < best_sse:
                    best_sse = sse
                    best_ideal_col = ideal_col
                    best_max_dev = float(np.max(np.abs(diff)))
            selection[train_col] = {"ideal_col": best_ideal_col, "sse": best_sse, "max_dev": best_max_dev}

        return selection
