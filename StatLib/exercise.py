# -*- coding: utf-8 -*-

import util.mystatutil as su
import sympy as sp
import numpy as np

def ex1(n: int=150, p: float=0.004, x: int=3):
    dist = su.PoissonDistribution(n * p)
    return 1 - sum([dist.prob(i) for i in range(x)])

def ex2(data):
    pass