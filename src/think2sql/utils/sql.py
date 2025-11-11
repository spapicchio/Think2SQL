import re
from collections import defaultdict
from functools import lru_cache

import duckdb

from think2sql.logger import get_logger

logger = get_logger(__name__)


def get_sql_from_generation(generation: str):
    # extract with regex everything is between <answer> and </answer>
    matches = re.findall(r"<answer>(.*?)</answer>", generation, re.DOTALL)
    if matches:
        # if matches found, return the last one without ```sql and ```
        return matches[-1].replace("```", "").replace("sql", "").strip()
    else:
        # if no matches found, extract everything between ```sql and ```
        matches = re.findall(r"```sql(.*?)```", generation, re.DOTALL)
        # if matches found, return the last one without ```sql and ```
        # otherwise return the original generation
        return matches[-1].strip() if matches else generation


def check_crud_sql(query):
    crud_keywords = {
        "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "TRUNCATE", "REPLACE",
        "GRANT", "REVOKE", "COMMIT", "ROLLBACK", "SAVEPOINT", "LOCK", "UNLOCK", "MERGE",
        "CALL", "EXECUTE", "EXPLAIN", "ANALYZE", "SET", "RENAME", "USE", "LOAD", "IMPORT",
        "EXPORT", "ATTACH", "DETACH", "VACUUM", "COMMENT", "SHUTDOWN", "PURGE", "OPTIMIZE"
    }
    query_upper = str(query).upper()
    for keyword in crud_keywords:
        # Use regex to match whole words only
        if re.search(r'\b' + re.escape(keyword) + r'\b', query_upper):
            return "CRUD_OPERATION_DETECTED"
    return query


@lru_cache(maxsize=512)
def extract_tbl2col_from_db(sqlite_path):
    con = duckdb.connect()
    con.execute(f"ATTACH '{sqlite_path}' AS mydb (TYPE SQLITE)")
    con.execute("SET search_path = mydb")
    # Query all columns grouped by schema and table
    rows = con.execute("""
                       SELECT table_schema, table_name, column_name
                       FROM information_schema.columns
                       ORDER BY table_schema, table_name, ordinal_position
                       """).fetchall()
    schema_map = defaultdict(list)
    for schema, table, col in rows:
        full_name = f"{table}" if schema else table
        schema_map[full_name].append(col)
    return schema_map


@lru_cache(maxsize=512)
def explain_query(sqlite_path, query):
    con = duckdb.connect()
    con.execute(f"ATTACH '{sqlite_path}' AS mydb (TYPE SQLITE)")
    con.execute("SET search_path = mydb")
    plan_json = con.execute(f"EXPLAIN (FORMAT JSON) {query}").fetchone()[1]
    return plan_json


def extract_tables_from_plan(plan_node, possible_tbls: set):
    tbl_found = set()
    if plan_node.get("extra_info", {}).get("Table", None):
        tbl = plan_node["extra_info"]["Table"]
        tbl_found.add(tbl)

    for child in plan_node.get("children", []):
        tbl_found.update(extract_tables_from_plan(child, possible_tbls))
    return tbl_found


def extract_cols_from_plan(plan_node, schema_map):
    cols_found = set()
    if plan_node.get("extra_info", {}).get("Projections", None):
        if 'Table' in plan_node["extra_info"]:
            tbl = plan_node["extra_info"]["Table"]
            avail_cols = {col.lower() for col in schema_map.get(tbl.lower(), [])}

            cols = plan_node["extra_info"]["Projections"]
            if isinstance(cols, str):
                cols = [cols]

            for col in cols:
                if not col.startswith('#') and col.lower() in avail_cols:
                    cols_found.add(f'{tbl}.{col}')

    for child in plan_node.get("children", []):
        cols_found.update(extract_cols_from_plan(child, schema_map))
    return cols_found


def extract_tables_and_columns_with_sqlglot(query, schema_map: dict, is_recursive=False) -> dict:
    import sqlglot
    from sqlglot import exp
    output = {'tables': set(), 'columns': set()}
    tables_in_schema = set(v.lower() for v in schema_map.keys())
    try:
        parsed = sqlglot.parse_one(query, dialect='sqlite')
        # Build a map alias -> table name
        alias_map = {}
        for t in parsed.find_all(exp.Table):
            if t.name in tables_in_schema:
                output['tables'].add(t.name)
                if t.alias:
                    alias_map[t.alias] = t.name
        # extract columns
        for col in parsed.find_all(exp.Column):
            # If there’s a table qualifier, use it
            table = col.table or ""
            if table == '' and col.parent_select and not is_recursive:
                query_parent = str(col.parent_select)
                output_ = extract_tables_and_columns_with_sqlglot(query_parent, schema_map, is_recursive=True)
                table = output_['tables'].pop() if output_['tables'] else ""
                if table == '':
                    # get all possible columns from tables in the
                    for tbl, possible_cols in schema_map.items():
                        if col.name in possible_cols:
                            table = tbl
                            break
            name = col.name
            resolved_table = alias_map.get(table, table) if table else ""
            cols_in_table = schema_map.get(resolved_table, [])
            cols_in_table = set(c.lower() for c in cols_in_table)
            if name in cols_in_table:
                qualified = f"{resolved_table}.{name}" if table else name
                output['columns'].add(qualified)
        return output
    except Exception as e:
        logger.warning(f'SQLglot not able to parse the query: {e}')
        return output
