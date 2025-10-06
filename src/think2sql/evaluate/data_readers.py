from typing import Protocol

from datasets import Dataset


class DataReader(Protocol):
    @staticmethod
    def read(file_path: str) -> Dataset:
        ...


class JsonDataReader:
    @staticmethod
    def read(file_path: str) -> Dataset:
        return Dataset.from_json(file_path)


class CsvDataReader:
    @staticmethod
    def read(file_path: str) -> Dataset:
        return Dataset.from_csv(file_path)


class ParquetDataReader:
    @staticmethod
    def read(file_path: str) -> Dataset:
        return Dataset.from_parquet(file_path)


class HFDataReader:
    @staticmethod
    def read(file_path: str, split='dev') -> Dataset:
        from datasets import load_dataset
        return load_dataset(file_path, split=split)
