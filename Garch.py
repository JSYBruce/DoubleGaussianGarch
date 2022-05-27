# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:47:15 2022

@author: BruceKing
"""
from typing import Optional, Union, cast

import numpy as np

from DoubleNormal import DoubleNormal

from arch.typing import Float64Array, Int32Array

from typing import Callable, List, Optional, Sequence, Tuple, Union

from util import garch_recursion

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

import itertools
_callback_info = {"iter": 0, "llf": 0.0, "count": 0, "display": 1}
class GARCH():
    r"""
    GARCH and related model estimation

    The following models can be specified using GARCH:
        * ARCH(p)
        * GARCH(p,q)
        * GJR-GARCH(p,o,q)
        * AVARCH(p)
        * AVGARCH(p,q)
        * TARCH(p,o,q)
        * Models with arbitrary, pre-specified powers

    Parameters
    ----------
    p : int
        Order of the symmetric innovation
    o : int
        Order of the asymmetric innovation
    q : int
        Order of the lagged (transformed) conditional variance
    power : float, optional
        Power to use with the innovations, abs(e) ** power.  Default is 2.0, which produces ARCH
        and related models. Using 1.0 produces AVARCH and related models.  Other powers can be
        specified, although these should be strictly positive, and usually larger than 0.25.

    Examples
    --------
    >>> from arch.univariate import GARCH

    Standard GARCH(1,1)

    >>> garch = GARCH(p=1, q=1)

    Asymmetric GJR-GARCH process

    >>> gjr = GARCH(p=1, o=1, q=1)

    Asymmetric TARCH process

    >>> tarch = GARCH(p=1, o=1, q=1, power=1.0)

    Notes
    -----
    In this class of processes, the variance dynamics are

    .. math::

        \sigma_{t}^{\lambda}=\omega
        + \sum_{i=1}^{p}\alpha_{i}\left|\epsilon_{t-i}\right|^{\lambda}
        +\sum_{j=1}^{o}\gamma_{j}\left|\epsilon_{t-j}\right|^{\lambda}
        I\left[\epsilon_{t-j}<0\right]+\sum_{k=1}^{q}\beta_{k}\sigma_{t-k}^{\lambda}
    """

    def __init__(self, p: int = 1, o: int = 0, q: int = 1, power: float = 2.0) -> None:
        self.p: int = int(p)
        self.o: int = int(o)
        self.q: int = int(q)
        self._normal = DoubleNormal()
        self.power: float = power
        self.name = "GARCH"
        self.num_params = 1 + p + o + q
        self._num_params = 1 + p + o + q
        if p < 0 or o < 0 or q < 0:
            raise ValueError("All lags lengths must be non-negative")
        if p == 0 and o == 0:
            raise ValueError("One of p or o must be strictly positive")
        if power <= 0.0:
            raise ValueError(
                "power must be strictly positive, usually larger than 0.25"
            )
        self._name = self._generate_name()

    def __str__(self) -> str:
        descr = self.name

        if self.power != 1.0 and self.power != 2.0:
            descr = descr[:-1] + ", "
        else:
            descr += "("

        for k, v in (("p", self.p), ("o", self.o), ("q", self.q)):
            if v > 0:
                descr += k + ": " + str(v) + ", "

        descr = descr[:-2] + ")"
        return descr

    def variance_bounds(self, resids: Float64Array, power: float = 2.0) -> Float64Array:
        """
        Construct loose bounds for conditional variances.

        These bounds are used in parameter estimation to ensure
        that the log-likelihood does not produce NaN values.

        Parameters
        ----------
        resids : ndarray
            Approximate residuals to use to compute the lower and upper bounds
            on the conditional variance
        power : float, optional
            Power used in the model. 2.0, the default corresponds to standard
            ARCH models that evolve in squares.

        Returns
        -------
        var_bounds : ndarray
            Array containing columns of lower and upper bounds with the same
            number of elements as resids
        """
        nobs = resids.shape[0]

        tau = min(75, nobs)
        w = 0.94 ** np.arange(tau)
        w = w / sum(w)
        var_bound = np.zeros(nobs)
        initial_value = w.dot(resids[:tau] ** 2.0)
        var_bounds = np.vstack((var_bound / 1e6, var_bound * 1e6)).T
        var = resids.var()
        min_upper_bound = 1 + (resids**2.0).max()
        lower_bound, upper_bound = var / 1e8, 1e7 * (1 + (resids**2.0).max())
        var_bounds[var_bounds[:, 0] < lower_bound, 0] = lower_bound
        var_bounds[var_bounds[:, 1] < min_upper_bound, 1] = min_upper_bound
        var_bounds[var_bounds[:, 1] > upper_bound, 1] = upper_bound

        if power != 2.0:
            var_bounds **= power / 2.0

        return np.ascontiguousarray(var_bounds)

    def _generate_name(self) -> str:
        p, o, q, power = self.p, self.o, self.q, self.power  # noqa: F841
        if power == 2.0:
            if o == 0 and q == 0:
                return "ARCH"
            elif o == 0:
                return "GARCH"
            else:
                return "GJR-GARCH"
        elif power == 1.0:
            if o == 0 and q == 0:
                return "AVARCH"
            elif o == 0:
                return "AVGARCH"
            else:
                return "TARCH/ZARCH"
        else:
            if o == 0 and q == 0:
                return "Power ARCH (power: {0:0.1f})".format(self.power)
            elif o == 0:
                return "Power GARCH (power: {0:0.1f})".format(self.power)
            else:
                return "Asym. Power GARCH (power: {0:0.1f})".format(self.power)

    def bounds(self, resids: Float64Array) -> List[Tuple[float, float]]:
        v = float(np.mean(abs(resids) ** self.power))

        bounds = [(1e-8 * v, 10.0 * float(v))]
        bounds.extend([(0.0, 1.0)] * self.p)
        for i in range(self.o):
            if i < self.p:
                bounds.append((-1.0, 2.0))
            else:
                bounds.append((0.0, 2.0))

        bounds.extend([(0.0, 1.0)] * self.q)
        bounds.extend([(0.0, 1.0)] * self.q)
        return bounds

    def constraints(self) -> Tuple[Float64Array, Float64Array]:
        p, o, q = self.p, self.o, self.q + 1
        k_arch = p + o + q
        # alpha[i] >0
        # alpha[i] + gamma[i] > 0 for i<=p, otherwise gamma[i]>0
        # beta[i] >0
        # sum(alpha) + 0.5 sum(gamma) + sum(beta) < 1
        a = np.zeros((k_arch + 2, k_arch + 1))
        for i in range(k_arch + 1):
            a[i, i] = 1.0
        for i in range(o):
            if i < p:
                a[i + p + 1, i + 1] = 1.0

        a[k_arch + 1, 1:] = -1.0
        a[k_arch + 1, p + 1 : p + o + 1] = -0.5
        b = np.zeros(k_arch + 2)
        b[k_arch + 1] = -1.0
        return a, b

    def compute_variance(
        self,
        parameters: Float64Array,
        resids: Float64Array,
        sigma1: Float64Array,
        sigma2: Float64Array,
        backcast: Union[float, Float64Array],
        var_bounds: Float64Array,
    ) -> Float64Array:
        # fresids is abs(resids) ** power
        # sresids is I(resids<0)
        power = self.power
        fresids = np.abs(resids) ** power
        sresids = np.sign(resids)

        p, o, q = self.p, self.o, self.q
        nobs = resids.shape[0]

        sigma1, sigma2 = garch_recursion(
            parameters, fresids, sresids, sigma1, sigma2, p, o, q, nobs, backcast, var_bounds
        )
        
      
        inv_power = 2.0 / power
        sigma2 **= inv_power

        return sigma1, sigma2

    def backcast(self, resids: Float64Array) -> Union[float, Float64Array]:
        power = self.power
        tau = min(75, resids.shape[0])
        w = 0.94 ** np.arange(tau)
        w = w / sum(w)
        backcast = np.sum((abs(resids[:tau]) ** power) * w)

        return float(backcast)
    
    def starting_values(self, resids: Float64Array) -> Float64Array:
        p, o, q = self.p, self.o, self.q + 1
        power = self.power
        alphas = [0.01, 0.05, 0.1, 0.2]
        gammas = alphas
        abg = [0.5, 0.7, 0.9, 0.98]
        abgs = list(itertools.product(*[alphas, gammas, abg]))
        target = np.mean(abs(resids) ** power)
        print(target)
        scale = np.mean(resids**2) / (target ** (2.0 / power))
        target *= scale ** (power / 2)

        svs = []
        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)
        llfs = np.zeros(len(abgs))
        for i, values in enumerate(abgs):
            alpha, gamma, agb = values
            sv = (1.0 - agb) * target * np.ones(p + o + q + 1)
            if p > 0:
                sv[1 : 1 + p] = alpha / p
                agb -= alpha
            if o > 0:
                sv[1 + p : 1 + p + o] = gamma / o
                agb -= gamma / 2.0
            if q > 0:
                sv[1 + p + o : 1 + p + o + q] = agb / q
            svs.append(sv)
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        loc = np.argmax(llfs)

        return svs[int(loc)]
    
    
    def _gaussian_loglikelihood(
        self,
        parameters: Float64Array,
        resids: Float64Array,
        backcast: Union[float, Float64Array],
        var_bounds: Float64Array,
    ) -> float:
        """
        Private implementation of a Gaussian log-likelihood for use in constructing starting
        values or other quantities that do not depend on the distribution used by the model.
        """
        sigma2 = np.zeros_like(resids)
        sigma1 = np.zeros_like(resids)
        self.compute_variance(parameters, resids, sigma1, sigma2, backcast, var_bounds)
        return float(self._normal.loglikelihood([], resids, sigma1, sigma2, 0))