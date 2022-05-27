# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:48:23 2022

@author: BruceKing
"""

from arch.compat.numba import jit

from abc import ABCMeta, abstractmethod
from typing import Optional, Union, cast

import numpy as np
from scipy.special import gammaln

from arch.typing import Float64Array, Int32Array
from arch.utility.array import AbstractDocStringInheritor

from typing import Callable, List, Optional, Sequence, Tuple, Union
import warnings

from numpy import (
    abs,
    array,
    asarray,
    empty,
    exp,
    int64,
    integer,
    isscalar,
    log,
    nan,
    ndarray,
    ones_like,
    pi,
    sign,
    sqrt,
    sum,
)
from numpy.random import Generator, RandomState, default_rng
from scipy.special import comb, gamma, gammainc, gammaincc, gammaln
import scipy.stats as stats

from arch.typing import ArrayLike, ArrayLike1D, Float64Array
from arch.utility.array import AbstractDocStringInheritor, ensure1d


def garch_recursion(
    parameters: Float64Array,
    fresids1: Float64Array,
    fresids2: Float64Array,
    sresids: Float64Array,
    sigma1: Float64Array,
    sigma2: Float64Array,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array,
) -> Float64Array:
    
    """
    Compute variance recursion for GARCH and related models

    Parameters
    ----------
    parameters : ndarray
        Model parameters
    fresids : ndarray
        Absolute value of residuals raised to the power in the model.  For
        example, in a standard GARCH model, the power is 2.0.
    sresids : ndarray
        Variable containing the sign of the residuals (-1.0, 0.0, 1.0)
    sigma2 : ndarray
        Conditional variances with same shape as resids
    p : int
        Number of symmetric innovations in model
    o : int
        Number of asymmetric innovations in model
    q : int
        Number of lags of the (transformed) variance in the model
    nobs : int
        Length of resids
    backcast : float
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        transformed variances for each time period
    """
    for t in range(nobs):
        loc = 0
        sigma1[t] = parameters[loc]
        sigma2[t] = parameters[loc]
        loc += 1
        for j in range(p):
            if (t - 1 - j) < 0:
                sigma1[t] += parameters[loc] * backcast
                sigma2[t] += parameters[loc+1] * backcast
            else:
                sigma1[t] += parameters[loc] * fresids1[t - 1 - j]
                sigma2[t] += parameters[loc+1] * fresids2[t - 1 - j]
            loc += 2
        for j in range(q):
            if (t - 1 - j) < 0:
                sigma1[t] += parameters[loc] * backcast
                sigma2[t] += parameters[loc+1] * backcast
            else:
                sigma1[t] += parameters[loc] * sigma1[t - 1 - j]
                sigma2[t] += parameters[loc+1] * sigma2[t - 1 - j]
            loc += 1
    return sigma1, sigma2
def format_float_fixed(x: float, max_digits: int = 10, decimal: int = 4) -> str:
    """Formats a floating point number so that if it can be well expressed
    in using a string with digits len, then it is converted simply, otherwise
    it is expressed in scientific notation"""
    # basic_format = '{:0.' + str(digits) + 'g}'
    if x == 0:
        return ("{:0." + str(decimal) + "f}").format(0.0)
    scale = np.log10(np.abs(x))
    scale = np.sign(scale) * np.ceil(np.abs(scale))
    if scale > (max_digits - 2 - decimal) or scale < -(decimal - 2):
        formatted = ("{0:" + str(max_digits) + "." + str(decimal) + "e}").format(x)
    else:
        formatted = ("{0:" + str(max_digits) + "." + str(decimal) + "f}").format(x)
    return formatted