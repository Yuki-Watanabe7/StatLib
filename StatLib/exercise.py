# -*- coding: utf-8 -*-

import util.mystatutil as su
import sympy as sp
import numpy as np
from typing import Any

def ex1(n: int=150, p: float=0.004, x: int=3):
    dist = su.PoissonDistribution(n * p)
    return 1 - sum([dist.prob(i) for i in range(x)])

def ex2(data: Any=np.array([1, 4, 5, 0, 3, 1, 4, 6, 2, 4])):
    return su.NegativeBinomialDistribution().estimate_param_by_unbiased(data)
