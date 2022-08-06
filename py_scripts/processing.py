from typing import List, Tuple
import pandas as pd


def adapt_to_group_lasso(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """Prepare dataframe to be fitted using a GroupLasso.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe. The columns of ``X`` shouldn't be one hot encoded.

    Returns
    -------
    X_dummies: pd.DataFrame
        X rearranged as continuous features, one-hot encoded
        categorical features.

    groups_size: list of int
        list of group size where continuous
        features are considered as group of size 1.

    Note
    ----
    This function changes the order of columns in dataframe.
    """
    X_rearranged = pd.concat((X.select_dtypes(exclude='object'),
                              X.select_dtypes(include='object')),
                             axis=1)
    groups_size = []
    for col in X_rearranged.columns:
        is_categorical = X[col].dtype == 'object'
        size = len(X[col].value_counts()) if is_categorical else 1
        groups_size.append(size)

    return pd.get_dummies(X_rearranged), groups_size


if __name__ == '__main__':
    pass
