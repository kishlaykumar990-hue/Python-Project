from src.exceptions import safe_read_csv, safe_save_to_db


class IdealLoader:
    """Loads ideal.csv (the 50 candidate functions) and saves it to the database."""

    def __init__(self, engine):
        """Store the shared database engine used to save the ideal functions."""
        self.engine = engine

    def load_ideal_csv(self, filepath):
        """Load ideal CSV file directly and rename its columns to X, Y1..Y50."""
        df = safe_read_csv(filepath, label="ideal CSV")
        if df is None:
            return None
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        n_functions = df.shape[1] - 1
        df.columns = ['X'] + [f"Y{i}" for i in range(1, n_functions + 1)]
        print(f"Ideal CSV loaded successfully from {filepath}. Columns: {list(df.columns)}")
        return df

    def save_to_db(self, df, table_name):
        """Save the ideal-functions DataFrame to the SQLite database."""
        safe_save_to_db(df, self.engine, table_name, label="ideal functions")
