from src.exceptions import safe_read_csv


class TestLoader:
    """Loads test.csv, ready for mapping against the selected ideal functions."""

    def load_test_csv(self, filepath):
        """Load test CSV file and remove unnamed columns."""
        df = safe_read_csv(filepath, label="test CSV")
        if df is None:
            return None
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        print(f"Test CSV loaded successfully from {filepath}. Columns: {list(df.columns)}")
        return df
