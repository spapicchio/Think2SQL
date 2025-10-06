import re


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
