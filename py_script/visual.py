from typing import List

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_categories(df: pd.DataFrame,
                    col_names: List,
                    n_cols=2,
                    height=800,
                    ) -> go.Figure:
    """Plot a bar chat of the class count of every category.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe.
    col_names : List
        Names of the categorical variables to consider.
    n_cols : int, optional
        Number of columns in subplot, by default 2
    height: int, optional
        height of the figure, by default 800

    Returns
    -------
    go.Figure
       subplot of (, n_cols) bar charts.
    """
    # assign (row, col) indices to each category
    n_cols = 3
    n_rows = sum(divmod(len(col_names), n_cols))
    dict_col_position = {col: {'row': divmod(i, n_cols)[0]+1,
                               'col': divmod(i, n_cols)[1]+1}
                         for i, col in enumerate(col_names)}

    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=col_names)

    for col, row_col in dict_col_position.items():
        classes_counts = df[col].value_counts()

        fig.add_trace(
            go.Bar(
                x=classes_counts.index,
                y=classes_counts.values,
                showlegend=False
            ), **row_col
        )

    fig.update_layout(title="Categories counts", height=height)
    return fig


if __name__ == '__main__':
    pass
