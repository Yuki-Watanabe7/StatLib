# -*- coding: utf-8 -*-

import util.mystatutil as su
import sympy as sp
import numpy as np
import pandas as pd
from typing import Any

def ex1(n: int=150, p: float=0.004, x: int=3):
    dist = su.PoissonDistribution(n * p)
    return 1 - sum([dist.prob(i) for i in range(x)])

def ex2(data: Any=np.array([1, 4, 5, 0, 3, 1, 4, 6, 2, 4])):
    return su.NegativeBinomialDistribution().estimate_param_by_unbiased(data)

def ex3(data: Any=su.CounterDict([(0, 49), (1, 88), (2, 118), (3, 57),
                                  (4, 28), (5, 16), (6, 4), (7, 3), (8, 2)]),
        pool: int=5, ):
    dist = su.PoissonDistribution()
    dist.lmd = dist.estimate_param_by_likelihood(data)[0][sp.Symbol('λ')]
    df = pd.DataFrame(data, index=['Actual']).T
    total = sum(data.values())
    df['Calced'] = [dist.prob(x) * total for x in df.index]
    df_pool = pd.DataFrame(df[df['Calced'] <= pool].sum(), columns=['Other']).T
    df_pool['Calced'] = total - df[df['Calced'] > pool]['Calced'].sum()
    df = pd.concat([df[df['Calced'] > pool], df_pool], axis=0)
    t = ((df['Actual'] - df['Calced']) ** 2 / df['Calced']).sum()
    return su.ChiSquaredDistribution(len(df) - 2).prob(t, sp.oo)