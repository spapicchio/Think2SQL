from typing import Set, List

import datasets
from datasets import DatasetDict, concatenate_datasets
from numba.cuda.initialize import initialize_all
from sqlglot.optimizer import qualify

from think2sql.configs import SFTScriptArguments
from think2sql.logger import get_logger

logger = get_logger(__name__)


def get_dataset(args: SFTScriptArguments, filter_fn=None) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    Args:
        args (SFTScriptArguments): Script arguments containing dataset configuration.

    Returns:
        DatasetDict: The loaded datasets.
    """
    if args.dataset_name.endswith(".json") or args.dataset_name.endswith(".csv"):
        logger.info(f"Loading dataset from local file: {args.dataset_name}")
        data_files = {"train": args.dataset_name}
        if args.dataset_mixture and args.dataset_mixture.test_split_size is not None:
            data_files["test"] = args.dataset_name  # Use the same file for test split
        dataset = datasets.load_dataset(
            "json" if args.dataset_name.endswith(".json") else "csv",
            data_files=data_files,
        )
        if filter_fn is not None:
            logger.info("Applying filter function to the dataset")
            initial_len = len(dataset)
            dataset = dataset.filter(filter_fn)
            logger.info(f"Filtered dataset from {initial_len} to {len(dataset)} examples")

        if args.add_test:
            dataset = dataset["train"].train_test_split(
                test_size=args.validation_split, shuffle=True, seed=2025
            )
        return dataset

    if args.dataset_name and not args.dataset_mixture:
        logger.info(f"Loading dataset: {args.dataset_name}")
        return datasets.load_dataset(args.dataset_name, args.dataset_config)

    elif args.dataset_mixture:
        logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")
        seed = args.dataset_mixture.seed
        datasets_list = []

        for dataset_config in args.dataset_mixture.datasets:
            logger.info(f"Loading dataset for mixture: {dataset_config.id} (config: {dataset_config.config})")
            ds = datasets.load_dataset(
                dataset_config.id,
                dataset_config.config,
                split=dataset_config.split,
            )
            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)
            if dataset_config.weight is not None:
                ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
                logger.info(
                    f"Subsampled dataset '{dataset_config.id}' (config: {dataset_config.config}) with weight={dataset_config.weight} to {len(ds)} examples"
                )

            datasets_list.append(ds)

        if datasets_list:
            combined_dataset = concatenate_datasets(datasets_list)
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

            if args.dataset_mixture.test_split_size is not None:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size, seed=seed
                )
                logger.info(
                    f"Split dataset into train and test sets with test size: {args.dataset_mixture.test_split_size}"
                )
                return combined_dataset
            else:
                return DatasetDict({"train": combined_dataset})
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")


# @lru_cache(maxsize=100)
# def utils_get_engine(relative_base_path,
#                      db_executor: AvailableDialect,
#                      db_id: str,
#                      cache=None,
#                      *args,
#                      **kwargs) -> BaseSQLDBExecutor:
#     try:
#         if db_executor == AvailableDialect.sqlite:
#             from NL2SQLEvaluator.db_executor.sqlite_executor import SqliteDBExecutor
#             relative_base_path =os.path.join(relative_base_path, db_id, f"{db_id}.sqlite")
#             engine = SqliteDBExecutor.from_uri(
#                 relative_base_path=relative_base_path, *args, **kwargs,
#             )
#             engine.cache_db = cache
#             return engine
#     except Exception as e:
#         raise ValueError(f"Error initializing database executor for {os.path.abspath('.')}, {relative_base_path}: {e}")
#

def utils_extract_table_names_sql(query: str, dialect: str = "sqlite") -> Set[str]:
    """
    Return table names referenced by a SQL query, excluding CTE names.
    Names are returned, possibly qualified as catalog.db.table if present in the query.
    """
    tree = sg.parse_one(query, read=dialect)
    # Collect CTE names so we don't mistake them for base tables
    cte_names = {cte.alias_or_name for cte in tree.find_all(E.CTE)}
    tables: Set[str] = set()
    for t in tree.find_all(E.Table):
        # Exclude references that point to a CTE alias
        if (t.name or "").lower() not in {n.lower() for n in cte_names}:
            tables.add(t.name)
    return tables


# pip install sqlglot sqlalchemy
from typing import Dict, Set, Optional
import sqlglot as sg
from sqlglot import exp as E


def extract_columns_sql(
        query: str,
        dialect: str = "sqlite",
        tbl2cols: Optional[Dict[str, Set[str]]] = None,
) -> List[str]:
    """
    Return a list of column references seen anywhere in the SQL.
    - Resolves table aliases to base tables.
    - Excludes CTE names as base tables.
    - For unqualified columns, resolves to their table using sqlglot's qualifier (per-scope).
    - Stars (*) returned as 'table.*' or '*' if unqualified (you can expand later with DB metadata).
    """
    # 1) Parse and qualify to resolve unqualified columns within each scope (CTEs, subqueries, main query)
    tree = sg.parse_one(query, read=dialect)
    # infer_aliases helps sqlglot assign names where needed
    qtree = qualify.qualify(tree, dialect=dialect, infer_aliases=True, )

    # 2) Collect CTE names to exclude from "real tables"
    cte_names: Set[str] = set()
    with_clause = qtree.args.get("with")
    if isinstance(with_clause, E.With):
        for cte in with_clause.expressions or []:
            # cte.alias will be an Identifier-like node; use .name if present
            alias_node = getattr(cte, "alias", None)
            alias_name = getattr(alias_node, "name", None) or getattr(alias_node, "this", None)
            if isinstance(alias_name, str):
                cte_names.add(alias_name.lower())

    # 3) Build alias -> base table map (skip CTE names)
    alias_to_table: Dict[str, str] = {}
    real_tables: Set[str] = set()

    for t in qtree.find_all(E.Table):
        tbl_name = t.name  # base table or CTE name
        if not tbl_name:
            continue
        alias_node = t.args.get("alias")
        # sqlglot represents alias in different ways; try to get a string
        alias_name = None
        if isinstance(alias_node, E.Identifier):
            alias_name = alias_node.this
        else:
            alias_name = getattr(alias_node, "name", None)
        alias_key = (alias_name or tbl_name).lower()

        # exclude CTE names from base tables
        if tbl_name.lower() in cte_names:
            # don't map alias to a CTE as "base table"
            continue

        real_tables.add(tbl_name)
        alias_to_table[alias_key] = tbl_name
        # Also map the table's own name to itself for convenient resolution
        alias_to_table[tbl_name.lower()] = tbl_name

    # 4) Stars in SELECT lists (SELECT * or SELECT t.*)
    result: List[str] = []
    tbl_with_stars: Set[Optional[str]] = set()
    for star in qtree.find_all(E.Star):
        tab_prefix = getattr(star, "table", None)
        if tab_prefix:
            # resolve alias to base table if possible
            resolved = alias_to_table.get(tab_prefix.lower())
            result.append(f"{resolved or tab_prefix}.*")
            tbl_with_stars.add(resolved or tab_prefix)
        else:
            result.append("*")
            tbl_with_stars.add(None)

    # 5) Regular columns
    for c in qtree.find_all(E.Column):
        tab = c.table  # after qualification, this should often be set
        col = c.name

        resolved_table = None
        if tab:
            resolved_table = alias_to_table.get(tab.lower(), tab)

        # If we already saw a star for this table, skip the individual columns
        if resolved_table in tbl_with_stars or (tab is None and None in tbl_with_stars):
            continue

        if resolved_table:
            result.append(f"{resolved_table}.{col}")
        else:
            # No table qualifier (still ambiguous after qualification) -> keep as plain column
            result.append(col)

    return result


if __name__ == "__main__":
    q = """
        WITH recent(u_id, last_t) AS (SELECT user_id, MAX(created_at) \
                                      FROM events \
                                      GROUP BY user_id)
        SELECT u.name, o.total, r.last_t
        FROM users AS u
                 JOIN orders o ON o.user_id = u.id
                 JOIN recent r ON r.u_id = u.id
        WHERE o.total > 100
        ORDER BY r.last_t DESC;
        """

    a = extract_columns_sql(q, dialect="sqlite")
    print(a)
