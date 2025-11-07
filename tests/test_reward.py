from think2sql.grpo.rewards import multi_tag_format_reward, reward_selected_tables, reward_selected_columns

example = example = """
    <reasoning>
The question asks: "How many stores with less need for products, and purchased through a distributor, are located in Washtenaw County?"

Step-by-step interpretation:

1. "Less need for products" → this maps to `Order Quantity = 1`. We find this in the `sales orders` table under column `order quantity`.

2. "Purchased through a distributor" → this means `Sales Channel = 'Distributor'`. In the schema, the column is named `sales channel`, so we must match that exactly.

3. "Located in Washtenaw County" → we look at the `store locations` table which has a `county` column. We filter where `county = 'washtenaw county'`.

4. To connect stores to sales orders, we use the foreign key `_storeid` in `sales orders` that references `storeid` in `store locations`.

So, we:
- Join `sales orders` with `store locations` on `_storeid = storeid`.
- Filter rows where:
   - `order quantity` = 1
   - `sales channel` = 'distributor'
   - `county` = 'washtenaw county'
- Count the number of such stores.

Note: The schema shows "Washtenaw County" in lowercase in the sample values (`washtenaw county`). So we must match it exactly as stored.

No aggregation beyond count is needed — we just want the total number of stores meeting all conditions.

Tables involved: `sales orders`, `store locations`
Columns used: 
- `sales orders.order quantity`, `sales orders.'sales channel'`, `_storeid`
- `store locations.county`, `store locations.storeid`

Join condition: `sales orders._storeid = store locations.storeid`

We do not need to include other tables like `customers`, `products`, or `sales team` because there’s no mention of them in the criteria.
</reasoning>

<tables>
sales orders, store locations
</tables>

<columns>
sales orders.`order quantity`, sales orders.`sales channel`, sales orders._storeid, store locations.county, store locations.storeid
</columns>

<checks>
1. All tables and columns exist in the schema — confirmed.
2. Join uses valid foreign key relationship: `_storeid` in `sales orders` references `storeid` in `store locations` — correct.
3. Filters match the given conditions: order quantity = 1, sales channel = 'distributor', county = 'washtenaw county'.
4. Query is read-only — yes.
5. Syntax is valid for SQLite — uses backticks for quoted identifiers, standard SELECT COUNT, JOIN, WHERE.
6. Case-sensitive comparison: the county value is stored as 'washtenaw county' (lowercase), so matching exactly ensures correctness.
</checks>

<answer>
SELECT COUNT(*) 
FROM `sales orders` 
JOIN `store locations` ON `sales orders`.`_storeid` = `store locations`.`storeid`
WHERE `sales orders`.`order quantity` = 1 
  AND `sales orders`.`sales channel` = 'distributor' 
  AND `store locations`.`county` = 'washtenaw county';
</answer>
    """


def test_multi_tag_format():
    assert multi_tag_format_reward([example]) == [1.0]


def test_tables_recall():
    assert reward_selected_tables([example], [['sales orders', 'store locations']]) == [1.0]


def test_column_recall():
    assert reward_selected_columns([example], [['sales orders.order quantity', 'sales orders.sales channel',
                                                'sales orders._storeid',
                                                'store locations.county', 'store locations.storeid']])
