from __future__ import division

import numpy as np


def __sum_squares(x):
    """
    Returns $\sum_i (x_i^2) - (\sum_i x_i)^2/n$
    """
    return np.sum(x**2) - np.sum(x)**2/len(x)


def __sum_products(x, y):
    """
    Returns $\sum_i (x_i y_i) - (\sum_i x_i \sum_i y_i)/n$
    """
    return np.sum(np.multiply(x,y)) - np.sum(x)*np.sum(y)/len(x)


def __sum_square_mean(x):
    """
    Returns the sum of the square of the difference from the
    mean of a distribution.

    Input
    =====
    :x, array-like: array

    Output
    ======
    $\sigma_i(x_i - \bar{x})^2$
    """
    x = np.asarray(x)
    return np.sum((x - np.mean(x))**2)


def __sum_square_residuals(yactual, ypredict):
    """
    Sum of the square of the residuals between actual and
    predicted values.

    Input
    =====
    :yactual, array-like: actual (measured) values
    :ypredict, array-like: predicted (model) values

    Output
    ======
    sum of the square of the residuals
    """
    yactual  = np.asarray(yactual)
    ypredict = np.asarray(ypredict)
    return np.sum((yactual - ypredict)**2)


def r_squared(yactual, ypredict):
    if len(yactual) != len(ypredict):
        msg = 'Lengths of vectors in do not match in call to R^2'
        raise ValueError(msg)
    # which is actual, which is predicted?
    if __sum_square_mean(yactual) < __sum_square_mean(ypredict):
        yactual, ypredict = ypredict, yactual
    SST = __sum_square_mean(yactual)
    SSR = __sum_square_residuals(yactual, ypredict)
    return 1. - SSR/SST


def covariance(yactual, ypredict):
    if len(yactual) != len(ypredict):
        msg = 'Lengths of vectors in do not match in call to covariance'
        raise ValueError(msg)
    rsq = r_squared(yactual, ypredict)
    if len(yactual) < 3:
        msg = 'Covariance cannot be calculated from fewer than ' \
              'three observations.'
        raise ValueError(msg)
    return 100.*np.sqrt((1/rsq - 1.)/(len(yactual) - 2))


def residual_variance(yactual, ypredict):
    if len(yactual) < 3:
        msg = 'Residual variance cannot be calculated from fewer than ' \
              'three observations'
        raise ValueError(msg)
    if len(yactual) != len(ypredict):
        msg = 'Lengths of vectors in do not match in call to residual variance'
        raise ValueError(msg)
    SSy = __sum_squares(yactual)
    SSx = __sum_squares(ypredict)
    SPxy = __sum_products(yactual, ypredict)
    n = len(yactual)
    return (SSy - SPxy/SSx)/(n-2)
