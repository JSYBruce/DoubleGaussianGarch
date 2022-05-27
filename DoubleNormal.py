# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from typing import Optional, Union, cast

import numpy as np
from scipy.special import gammaln

from arch.typing import Float64Array, Int32Array
from arch.utility.array import AbstractDocStringInheritor

from typing import Callable, List, Optional, Sequence, Tuple, Union

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

from arch.typing import ArrayLike, ArrayLike1D, Float64Array

class DoubleNormal():
    """
    Standard normal distribution for use with ARCH models

    Parameters
    ----------
    """

    def __init__(
        self) -> None:
        self._name = "Double_Normal"
        self.name = "Double_Normal"
    def constraints(self) -> Tuple[Float64Array, Float64Array]:
        return empty(0), empty(0)

    def bounds(self, resids: Float64Array) -> List[Tuple[float, float]]:
        return []

    def loglikelihood(
        self,
        parameters: Union[Sequence[float], ArrayLike1D],
        resids: ArrayLike,
        sigma1: ArrayLike,
        sigma2: ArrayLike,
        weight: Float64Array,
        individual: bool = False,
    ) -> Union[float, Float64Array]:
        r"""Computes the log-likelihood of assuming residuals are normally
        distributed, conditional on the variance

        Parameters
        ----------
        parameters : ndarray
            The normal likelihood has no shape parameters. Empty since the
            standard normal has no shape parameters.
        resids  : ndarray
            The residuals to use in the log-likelihood calculation
        sigma2 : ndarray
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \ln f\left(x\right)=-\frac{1}{2}\left(\ln2\pi+\ln\sigma^{2}
            +\frac{x^{2}}{\sigma^{2}}\right)

        """
        lls = log( weight/sqrt(2*pi*sigma1)*exp(-0.5*resids**2/sigma1) + (1-weight)/sqrt(2*pi*sigma2)*exp(-0.5*resids**2/sigma2) )
        if individual:
            return lls
        else:
            return sum(lls)

    def starting_values(self, std_resid: Float64Array) -> Float64Array:
        return empty(0)
    