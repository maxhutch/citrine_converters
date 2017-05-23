from __future__ import division

import numpy as np
from bisect import bisect_left as bisect

def resample(arr, **kwds):
    """
    Resample the array-like object based on intensity.

    Input
    =====
    :arr, array-like: object to be resampled

    Options
    =======
    :num, int: number of samples to use in resampling `arr`

    Output
    ======
    Resampled array
    """
    arr = np.asarray(arr)
    num = kwds.get('num', arr.size)
    # resample
    resampled = np.zeros_like(arr)
    resampled_r = resampled.ravel()
    # construct continuous distribution function
    cdf = np.cumsum(arr.ravel())
    cdf = (cdf - cdf.min())/(cdf.max() - cdf.min())
    for _ in xrange(arr.size):
        i = bisect(cdf, np.random.random())
        resampled_r[i] += 1
    return resampled
