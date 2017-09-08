import numpy as np
from scipy.stats import iqr

def remove_outliers(vec):
    """
    Removes outliers based on 1.5*IQR

    :param vec:
    :return:
    """
    mask = np.ones_like(vec, dtype=bool)
    for _ in range(len(vec)):
        mean = np.mean(vec[mask])
        inner_quartile = iqr(vec[mask])
        iqr_distance = np.abs(vec[mask] - mean)
        if np.max(iqr_distance) > 1.5*inner_quartile:
            index = np.argmax(iqr_distance)
            tmp = mask[mask]
            tmp[index] = False
            mask[mask] = tmp
        else:
            return mask