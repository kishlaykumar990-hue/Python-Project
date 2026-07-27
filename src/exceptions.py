import pandas as pd
from sqlalchemy.exc import SQLAlchemyError


class MappingErrorHandling(Exception):
    """Raised when test-to-ideal mapping produces zero successful matches."""
    pass


def safe_read_csv(filepath, label="CSV"):
    """Read a CSV file, handling standard exceptions in one place.

    Returns the loaded DataFrame, or None if the file could not be read.
    Used by the loader classes so every CSV-reading method shares the same
    error-handling logic instead of each repeating its own try/except.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Unexpected error loading {label}: {e}")
        return None


def safe_save_to_db(df, engine, table_name, label="data"):
    """Save a DataFrame to the SQLite database, handling standard
    exceptions in one place. Used by every class that writes to the
    database, instead of each repeating its own try/except.
    """
    try:
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
        print(f"Table '{table_name}' created and populated with {label}.")
    except SQLAlchemyError as e:
        print(f"SQLAlchemy error storing table '{table_name}': {e}")
    except Exception as e:
        print(f"Unexpected error storing table '{table_name}': {e}")


def safe_load_table_from_db(engine, table_name):
    """Load a table from the SQLite database, handling standard exceptions
    in one place.
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", con=conn)
            print(f"Loaded table '{table_name}' successfully. Columns: {list(df.columns)}")
            return df
    except SQLAlchemyError as e:
        print(f"SQLAlchemy error loading table '{table_name}': {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading table '{table_name}': {e}")
        return None
