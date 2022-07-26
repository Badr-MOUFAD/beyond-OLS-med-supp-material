from typing import List, Tuple

import pandas as pd
import numpy as np
from celer import celer_path

from py_scripts.processing import adapt_to_group_lasso


def score_features(X: pd.DataFrame,
                   y: pd.DataFrame
                   ) -> List[Tuple[str, float]]:
    r"""Score features in dataframe using GroupLasso.

    Score function:

    .. math::

        \max \{ \frac{\lambda}{\lambda_{max}} \ | \ \beta_j \neq 0 \}


    Parameters
    ----------
    X : pd.DataFrame
        Dataframe. Feature with ``object`` type are considered categorical.

    y : pd.DataFrame
        Target feature.

    Returns
    -------
    np.ndarray
        Return a list of tuples with the name of feature and the corresponding score.
        ``[(feature_name, score), ...]``.
    """
    X_with_dummies, groups = adapt_to_group_lasso(X)

    n_groups = len(groups)
    grp_scores = np.zeros(n_groups)
    grp_indices = np.append([0], np.cumsum(groups))

    n_alphas = 100
    alphas, coefs, _ = celer_path(X_with_dummies.values, y.values.flatten(),
                                  pb='grouplasso', groups=groups,
                                  n_alphas=n_alphas, eps=1e-6, prune=1)

    for i in range(n_alphas):
        for g in range(n_groups):
            # skip groups whose scores are already computed
            if grp_scores[g] != 0:
                continue

            # select group coefs
            grp_g_indices = slice(grp_indices[g], grp_indices[g+1])
            coefs_g = coefs[:, i][grp_g_indices]

            if np.linalg.norm(coefs_g, ord=np.inf) > 0:
                grp_scores[g] = alphas[i] / alphas[0]  # alpha / alpha_max

    return list(zip(X.columns, grp_scores))


if __name__ == '__main__':
    pass
