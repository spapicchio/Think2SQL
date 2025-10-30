# from typing import Optional, Literal
#
# # from NL2SQLEvaluator.orchestrator_state import AvailableDialect
#
# from think2sql.utils.data import utils_get_engine
#
#
# def get_schema(row, add_sample_rows_strategy: Optional[Literal["append", "inline"]], relative_base_path: str):
#     engine = utils_get_engine(relative_base_path, AvailableDialect('sqlite'), row['db_id'])
#     return {"schema": engine.get_ddl_database(add_sample_rows_strategy=add_sample_rows_strategy)}
#
