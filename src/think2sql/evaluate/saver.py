from pathlib import Path
from typing import Protocol

from pandas import DataFrame

from think2sql.logger import get_logger

logger = get_logger(__name__)


class DataframeSaver(Protocol):
    def save(self, folder, file_name, df: DataFrame, *args, **kwargs):
        """Save a file with the given name and parameters."""
        pass


class JSONSaver():
    def save(self, folder, file_name, df: DataFrame, *args, **kwargs):
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        path = Path(folder) / date_str / f"{file_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(path, orient='records', indent=4,)
        logger.info(f"Saved DF in {path} as JSON")

