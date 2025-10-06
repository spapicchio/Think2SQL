import json
from pathlib import Path
from typing import Protocol, Any

from pandas import DataFrame

from think2sql.logger import get_logger

logger = get_logger(__name__)


class DataframeSaver(Protocol):
    @staticmethod
    def save(folder, df: DataFrame, configs: tuple[Any], *args, **kwargs):
        """Save a file with the given name and parameters."""
        pass


class JSONSaver:
    @staticmethod
    def save(folder: Path, df: DataFrame, configs: tuple, *args, **kwargs):
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        path = Path(folder) / date_str / "df.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(path, orient='records', indent=4, )
        logger.info(f"Saved DF in {path} as JSON")
        for config in configs:
            config_name = getattr(config, "__name__", config.__class__.__name__)
            config_path = path.parent / f"{config_name}.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4, default=str)
            logger.info(f"Saved config in {config_path} as JSON")
