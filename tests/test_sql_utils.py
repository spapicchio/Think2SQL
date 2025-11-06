# ---------------------------------------
# Extra fixture: a richer SQLite database
# ---------------------------------------

import json
import sqlite3

import pytest

from think2sql.utils.sql import (
    extract_tbl2col_from_db,
    explain_query,
    extract_tables_from_plan,
    extract_cols_from_plan,
)


@pytest.fixture
def sqlite_db_complex(tmp_path):
    """
    Creates a more complete retail-like schema:

      customers(id INTEGER PK, name TEXT, city TEXT)
      orders(id INTEGER PK, customer_id INTEGER, order_date TEXT, total REAL)
      order_items(order_id INTEGER, product_id INTEGER, qty INTEGER, price REAL)
      products(id INTEGER PK, category_id INTEGER, name TEXT)
      categories(id INTEGER PK, name TEXT)

    Returns the path to the SQLite file.
    """
    db_path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.executescript("""
        PRAGMA foreign_keys = OFF;

        CREATE TABLE customers(
            id INTEGER PRIMARY KEY,
            name TEXT,
            city TEXT
        );

        CREATE TABLE orders(
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date TEXT,
            total REAL
        );

        CREATE TABLE order_items(
            order_id INTEGER,
            product_id INTEGER,
            qty INTEGER,
            price REAL
        );

        CREATE TABLE products(
            id INTEGER PRIMARY KEY,
            category_id INTEGER,
            name TEXT
        );

        CREATE TABLE categories(
            id INTEGER PRIMARY KEY,
            name TEXT
        );
    """)

    customers = [
        (1, "Alice", "Turin"),
        (2, "Bob", "Milan"),
        (3, "Carla", "Rome"),
        (4, "Dan", "Turin"),
    ]
    cur.executemany("INSERT INTO customers(id, name, city) VALUES (?, ?, ?)", customers)

    categories = [
        (10, "Electronics"),
        (20, "Books"),
        (30, "Groceries"),
    ]
    cur.executemany("INSERT INTO categories(id, name) VALUES (?, ?)", categories)

    products = [
        (100, 10, "Adapter"),
        (101, 10, "Audio Cable"),
        (200, 20, "Algorithms Book"),
        (300, 30, "Apples"),
    ]
    cur.executemany("INSERT INTO products(id, category_id, name) VALUES (?, ?, ?)", products)

    orders = [
        (1000, 1, "2025-01-02", 129.90),
        (1001, 1, "2025-02-11", 49.99),
        (1002, 2, "2025-03-05", 15.50),
        (1003, 3, "2025-04-15", 250.00),
        # Customer 4 has no orders
    ]
    cur.executemany("INSERT INTO orders(id, customer_id, order_date, total) VALUES (?, ?, ?, ?)", orders)

    order_items = [
        (1000, 100, 1, 19.90),
        (1000, 101, 2, 15.00),
        (1001, 200, 1, 49.99),
        (1002, 300, 3, 5.00),
        (1003, 100, 5, 19.00),
        (1003, 200, 1, 55.00),
    ]
    cur.executemany("INSERT INTO order_items(order_id, product_id, qty, price) VALUES (?, ?, ?, ?)", order_items)

    con.commit()
    con.close()
    return str(db_path)


# ---------------------------------------
# Helper to normalize plan result (string or dict/list)
# ---------------------------------------
def _normalize_plan(plan):
    if isinstance(plan, str):
        return json.loads(plan)
    return plan


# ---------------------------------------------------------
# 1) CTE + window + GROUP BY + UNION ALL + ORDER BY
# ---------------------------------------------------------
def test_real_plan_cte_window_union(sqlite_db_complex):
    schema_map = extract_tbl2col_from_db(sqlite_db_complex)

    query = """
            WITH oi AS (SELECT oi.order_id, (oi.qty * oi.price) AS line_total
                        FROM order_items oi),
                 cust_totals AS (SELECT o.customer_id,
                                        SUM(oi.line_total)                                                              AS revenue,
                                        ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY SUM(oi.line_total) DESC) AS rn
                                 FROM orders o
                                          JOIN oi ON oi.order_id = o.id
                                 GROUP BY o.customer_id)
            SELECT c.name, ct.revenue
            FROM customers c
                     JOIN cust_totals ct ON ct.customer_id = c.id
            WHERE ct.rn = 1
            UNION ALL
            SELECT c.name, 0.0 AS revenue
            FROM customers c
                     LEFT JOIN orders o ON o.customer_id = c.id
            WHERE o.id IS NULL
            ORDER BY revenue DESC, name ASC \
            """

    plan = _normalize_plan(explain_query(sqlite_db_complex, query))

    # Collect from the full tree
    tables = set()
    cols = set()
    for node in plan:
        tables |= extract_tables_from_plan(node, possible_tbls=set(schema_map.keys()))
        cols |= extract_cols_from_plan(node, schema_map)

    # Must see the core tables touched
    assert {"customers", "orders", "order_items"} == tables

    # Columns that should appear in the plan (subset, not exact match)
    expected_cols_subset = {
        "customers.name",
        "customers.id",

        "orders.id",
        "orders.customer_id",

        "order_items.order_id",
        "order_items.qty",
        "order_items.price",
    }
    assert expected_cols_subset == cols


# ---------------------------------------------------------
# 2) EXISTS (correlated subquery) + predicate on aggregate
# ---------------------------------------------------------
def test_real_plan_exists_correlated(sqlite_db_complex):
    schema_map = extract_tbl2col_from_db(sqlite_db_complex)

    query = """
            SELECT c.id, c.name
            FROM customers c
            WHERE EXISTS (SELECT 1
                          FROM orders o
                          WHERE o.customer_id = c.id
                            AND o.total > 50)
            ORDER BY c.id \
            """
    plan = _normalize_plan(explain_query(sqlite_db_complex, query))

    tables = set()
    cols = set()
    for node in plan:
        tables |= extract_tables_from_plan(node, possible_tbls=set(schema_map.keys()))
        cols |= extract_cols_from_plan(node, schema_map)

    # EXISTS should bring in orders
    assert {"customers", "orders"} == tables
    assert {"customers.id", "customers.name", "orders.customer_id", "orders.total"} == cols


# ---------------------------------------------------------
# 3) DISTINCT + LIMIT + quoted identifiers + LEFT JOIN
# ---------------------------------------------------------
def test_real_plan_distinct_limit_quoted(sqlite_db_complex):
    schema_map = extract_tbl2col_from_db(sqlite_db_complex)

    query = """
            SELECT DISTINCT p.name
            FROM products p
                     LEFT JOIN categories "cat" ON p.category_id = "cat".id
            WHERE ("cat".name IS NOT NULL OR p.name LIKE 'A%')
            ORDER BY p.name
            LIMIT 5 \
            """
    plan = _normalize_plan(explain_query(sqlite_db_complex, query))

    tables = set()
    cols = set()
    for node in plan:
        tables |= extract_tables_from_plan(node, possible_tbls=set(schema_map.keys()))
        cols |= extract_cols_from_plan(node, schema_map)

    assert {"products", "categories"} == tables
    assert {"products.name", "products.category_id", "categories.id", "categories.name"} == cols

    # ---------------------------------------------------------
    # 4) GROUP BY + HAVING (with COUNT/DISTINCT) + LEFT JOIN
    # ---------------------------------------------------------


def test_real_plan_groupby_having(sqlite_db_complex):
    schema_map = extract_tbl2col_from_db(sqlite_db_complex)

    query = """
            SELECT c.city, COUNT(DISTINCT o.id) AS num_orders
            FROM customers c
                     LEFT JOIN orders o ON o.customer_id = c.id
            GROUP BY c.city
            HAVING COUNT(*) >= 1
            ORDER BY num_orders DESC, c.city ASC \
            """
    plan = _normalize_plan(explain_query(sqlite_db_complex, query))

    tables = set()
    cols = set()
    for node in plan:
        tables |= extract_tables_from_plan(node, possible_tbls=set(schema_map.keys()))
        cols |= extract_cols_from_plan(node, schema_map)

    assert {"customers", "orders"} == tables
    assert {"customers.city", "orders.id", "orders.customer_id", 'customers.id'} == cols


# ---------------------------------------------------------
# 5) CROSS JOIN + scalar subquery in SELECT-list
# ---------------------------------------------------------
def test_real_plan_cross_join_scalar_subquery(sqlite_db_complex):
    schema_map = extract_tbl2col_from_db(sqlite_db_complex)

    query = """
            SELECT p.name,
                   (SELECT COUNT(price) FROM order_items) AS avg_price
            FROM products p
                     CROSS JOIN (SELECT 1 AS one) t
            ORDER BY p.name \
            """
    plan = _normalize_plan(explain_query(sqlite_db_complex, query))

    tables = set()
    cols = set()
    for node in plan:
        tables |= extract_tables_from_plan(node, possible_tbls=set(schema_map.keys()))
        cols |= extract_cols_from_plan(node, schema_map)

    assert {"products", "order_items"} == tables
    assert {"products.name", "order_items.price"} == cols
