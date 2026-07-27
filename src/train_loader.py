from src.exceptions import safe_read_csv, safe_save_to_db


class TrainLoader:
    """Loads train.csv and saves the training data to the database."""

    def __init__(self, engine):
        """Store the shared database engine used to save the training data."""
        self.engine = engine

    def load_train_csv(self, filepath):
        """Load train CSV file and remove unnamed columns."""
        df = safe_read_csv(filepath, label="train CSV")
        if df is None:
            return None
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        print(f"Train CSV loaded successfully from {filepath}. Columns: {list(df.columns)}")
        return df

    def save_to_db(self, df, table_name):
        """Save the training DataFrame to the SQLite database."""
        safe_save_to_db(df, self.engine, table_name, label="training data")
