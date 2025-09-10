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
