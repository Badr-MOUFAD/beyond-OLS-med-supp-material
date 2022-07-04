from typing import List, Tuple
import pandas as pd


def adapt_to_group_lasso(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """Prepare dataframe for GroupLasso.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe.

    Returns
    -------
    Tuple[pd.DataFrame, List[int]]
        Returns X rearranged as numerical features, one-hot encoded
        categorical features and a list of group size (every numerical
        feature is considered as single group).

    Note
    ----
    This function will changes the order of columns dataframe.
    """
    X_rearranged = pd.concat((X.select_dtypes(exclude='object'),
                              X.select_dtypes(include='object')),
                             axis=1)
    groups_size = []
    for col in X_rearranged.columns:
        is_categorical = X[col].dtype == 'object'
        size = len(X[col].value_counts()) if is_categorical else 1
        groups_size.append(size)

    return (pd.get_dummies(X_rearranged), groups_size)


if __name__ == '__main__':
    pass
