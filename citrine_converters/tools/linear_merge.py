from __future__ import division

import numpy as np

def linear_merge(x1, y1, x2, y2):
    """
    Merge data pairs (x1, y1) and (x2, y2) over the intersection
    of their ranges (x) using a linear interpolation of each
    dataset to interpolate between unaligned values.

    No extrapolation is performed.

    Input
    =====
    :x1, array-like: abscissa coordinates of dataset 1
    :y1, array-like: ordinate coordinates of dataset 1
    :x2, array-like: abscissa coordinates of dataset 2
    :y2, array-like: ordinate coordinates of dataset 2

    OUT
    ===
    Tuple of the merged x (`xm`) and interpolated y1 (`y1m`) and y2 (`y2m`):
    `(xm, y1m, y2m)`
    """
    # ensure all values are ndarrays
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    # ##########
    # merge on x
    xmerge = np.concatenate((np.sort(x1), np.sort(x2)))
    xmerge.sort(kind='mergesort')
    # perform interpolation
    xlo = np.max((x1.min(), x2.min()))
    xhi = np.min((x1.max(), x2.max()))
    # keep only the merged x values within the intersection of
    # the data ranges
    mask = (xmerge >= xlo) & (xmerge <= xhi)
    xf = xmerge[mask]
    y1f = np.interp(xf, x1, y1)
    y2f = np.interp(xf, x2, y2)
    # return the interpolated, merged dataset
    return (xf, y1f, y2f)
