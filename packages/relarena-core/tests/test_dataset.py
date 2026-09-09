"""Shared split-table operations."""

import pandas as pd
from relbench.base import Table

from relarena_core.dataset import concat_tables


def test_concat_tables_unions_rows_and_keeps_schema() -> None:
    a = Table(
        df=pd.DataFrame({"entity": [1, 2], "t": [10, 11], "y": [0.0, 1.0]}),
        fkey_col_to_pkey_table={"entity": "users"},
        pkey_col=None,
        time_col="t",
    )
    b = Table(
        df=pd.DataFrame({"entity": [3], "t": [12], "y": [2.0]}),
        fkey_col_to_pkey_table={"entity": "users"},
        pkey_col=None,
        time_col="t",
    )
    c = concat_tables(a, b)
    assert len(c.df) == 3
    assert list(c.df["y"]) == [0.0, 1.0, 2.0]
    assert c.time_col == "t"
    assert c.fkey_col_to_pkey_table == {"entity": "users"}
    # inputs are untouched
    assert len(a.df) == 2 and len(b.df) == 1
