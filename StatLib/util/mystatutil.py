# -*- coding: utf-8 -*-

import sympy as sp
import numpy as np
import functools as ft
from typing import Any, Dict

def arg_check(arg_type=object, min_value=-sp.oo, min_include=True,
              max_value=sp.oo, max_include=True):
    def _arg_check(func):
        @ft.wraps(func)
        def __arg_check(*args):
            arg = args[1]
            if not isinstance(arg, sp.Symbol):
                if not isinstance(arg, arg_type):
                    raise TypeError('{} is expected, but {}.'.format(arg_type, type(arg)))
                if arg < min_value:
                    if not (min_include and arg == min_value):
                        raise ValueError('Min is {}, but {}.'.format(min_value, arg))
                if max_value < arg:
                    if not (max_include and arg == max_value):
                        raise ValueError('Max is {}, but {}.'.format(max_value, arg))
            return func(*args)
        return __arg_check
    return _arg_check

def param(param_name, param_symbol, param_type=object,
          min_value=-sp.oo, min_include=True, max_value=sp.oo, max_include=True):
    def _param(klass):
        @property
        def parameter(self) -> Any:
            return self._params[sp.Symbol(param_symbol)]

        @parameter.setter
        @arg_check(param_type, min_value, min_include, max_value, max_include)
        def parameter(self, param_value) -> None:
            self._params[sp.Symbol(param_symbol)] = param_value

        @parameter.deleter
        def parameter(self) -> None:
            self._params[sp.Symbol(param_symbol)] = sp.Symbol(param_symbol)

        setattr(klass, param_name, parameter)
        klass.param_validity[sp.Symbol(param_symbol)] = {'param_name': param_name,
                                                         'param_type': param_type,
                                                         'min_value': min_value,
                                                         'min_include': min_include,
                                                         'max_value': max_value,
                                                         'max_include': max_include}
        return klass
    return _param

def param_p_q(klass):
    @property
    def p(self) -> Any:
        return self._params[sp.Symbol('p')]

    @p.setter
    @arg_check(min_value=0, max_value=1)
    def p(self, p: Any) -> None:
        self._params[sp.Symbol('p')] = p

    @p.deleter
    def p(self) -> None:
        self._params[sp.Symbol('p')] = sp.Symbol('p')

    @property
    def q(self) -> Any:
        return 1 - self._params[sp.Symbol('p')]

    @q.setter
    @arg_check(min_value=0, max_value=1)
    def q(self, q) -> None:
        self._params[sp.Symbol('p')] = 1 - q

    @q.deleter
    def q(self) -> None:
        self._params[sp.Symbol('p')] = sp.Symbol('p')

    @property
    def pdf(self) -> Any:
        return self._PDF.subs(self._params).subs(sp.Symbol('q'), self.q)

    setattr(klass, 'p', p)
    setattr(klass, 'q', q)
    setattr(klass, 'pdf', pdf)
    klass.param_validity[sp.Symbol('p')] = {'param_name': 'p',
                                            'param_type': object,
                                            'min_value': 0,
                                            'min_include': True,
                                            'max_value': 1,
                                            'max_include': True}
    return klass

def param_as_min_max(param_name, param_symbol, min_max,
                     param_type=object, min_value=-sp.oo, min_include=True,
                     max_value=sp.oo, max_include=True):
    def _param(klass):
        @property
        def parameter(self) -> Any:
            return self._params[sp.Symbol(param_symbol)]

        @parameter.setter
        @arg_check(param_type, min_value, min_include, max_value, max_include)
        def parameter(self, param_value) -> None:
            self._params[sp.Symbol(param_symbol)] = param_value
            setattr(self, min_max, param_value)

        @parameter.deleter
        def parameter(self) -> None:
            self._params[sp.Symbol(param_symbol)] = sp.Symbol(param_symbol)
            setattr(self, min_max, sp.Symbol(param_symbol))

        setattr(klass, param_name, parameter)
        klass.param_validity[sp.Symbol(param_symbol)] = {'param_name': param_name,
                                                         'param_type': param_type,
                                                         'min_value': min_value,
                                                         'min_include': min_include,
                                                         'max_value': max_value,
                                                         'max_include': max_include}
        return klass
    return _param

class ProbabilityDistribution():

    param_validity = {}

    def __init__(self) -> None:
        self._params = {}
        self.x_min = -sp.oo
        self.x_max = sp.oo

    @property
    def pdf(self) -> Any:
        return self._PDF.subs(self._params)

    def prob(self, x_from: Any, x_to: Any) -> Any:
        _z = sp.Symbol('z')
        return self.cdf.subs(_z, x_to) - self.cdf.subs(_z, x_from)

    def moment(self, degree: int) -> Any:
        _t = sp.Symbol('t')
        return sp.diff(self.mgf, _t, degree).subs(_t, 0)

    @property
    def expected_value(self) -> Any:
        return self.moment(1)

    @property
    def variance(self) -> Any:
        return sp.factor(self.moment(2) - self.expected_value ** 2)

    def estimate_param_by_moment(self, data: Any):
        unknown_params = [key for key, value in self._params.items()
                          if key == value]
        eq_list = [self.moment(d) - (data ** d).mean()
                   for d in range(1, len(unknown_params) + 1)]
        ans = sp.solve(eq_list, *unknown_params)
        candidates = [ans] if isinstance(ans, dict) else [dict(zip(unknown_params, a))
                                                          for a in ans]
        rtn = []
        for candidate in candidates:
            for key, value in candidate.items():
                if not self.is_valid_param(key, value):
                    break
            else:
                rtn.append(candidate)
        return rtn

    def estimate_param_by_unbiased(self, data: Any):
        unknown_params = [key for key, value in self._params.items()
                          if key == value]
        if len(unknown_params) == 1:
            eq_list = [self.expected_value - data.mean()]
        elif len(unknown_params) == 2:
            eq_list = [self.expected_value - data.mean(), self.variance - data.var(ddof=1)]
        if len(unknown_params) > 2:
            raise NotImplementedError('Unknow Parameters must be less than 2.')
        ans = sp.solve(eq_list, *unknown_params)
        candidates = [ans] if isinstance(ans, dict) else [dict(zip(unknown_params, a))
                                                          for a in ans]
        rtn = []
        for candidate in candidates:
            for key, value in candidate.items():
                if not self.is_valid_param(key, value):
                    break
            else:
                rtn.append(candidate)
        return rtn

    def reset_params(self) -> None:
        for param in self._params.keys():
            self._params[param] = param

    def is_valid_param(self, symbol, value) -> bool:
        validity = self.param_validity[symbol]
        try:
            arg_check(validity['param_type'], validity['min_value'], validity['min_include'],
                      validity['max_value'], validity['max_include'])(lambda x, y: y)(self, value)
            return True
        except:
            return False

class DiscreteProbabilityDistribution(ProbabilityDistribution):

    param_validity = {}

    def __init__(self) -> None:
        super().__init__()
        self.x_min = 0

    @property
    def cdf(self) -> Any:
        return sp.summation(self.pdf, (sp.Symbol('x'), self.x_min, sp.Symbol('z')))

    @property
    def mgf(self) -> Any:
        _t, _x = sp.symbols('t x')
        return sp.summation(sp.exp(_t * _x) * self.pdf, (sp.Symbol('x'), self.x_min, self.x_max))

    @arg_check(int, min_value=0)
    def prob(self, x: int) -> Any:
        return self.pdf.subs(sp.Symbol('x'), x)

class ContinuousProbabilityDistribution(ProbabilityDistribution):

    param_validity = {}

    def __init__(self) -> None:
        super().__init__()
    
    @property
    def cdf(self) -> Any:
        return sp.integrate(self.pdf, (sp.Symbol('x'), self.x_min, sp.Symbol('z')))

    @property
    def mgf(self) -> Any:
        _t, _x = sp.symbols('t x')
        return sp.integrate(sp.exp(_t * _x) * self.pdf, (sp.Symbol('x'), self.x_min, self.x_max))

@param_as_min_max('n', 'n', 'x_max', param_type=int, min_value=1)
@param_p_q
class BinomialDistribution(DiscreteProbabilityDistribution):

    param_validity = {}
    _n, _p, _q, _x = sp.symbols('n p q x')
    _PDF = sp.binomial(_n, _x) * _p ** _x * _q ** (_n - _x)

    def __init__(self, n: Any=sp.Symbol('n'), p: Any=sp.Symbol('p')) -> None:
        super().__init__()
        self.n = n
        self.p = p

    @property
    def mgf(self) -> Any:
        ''' 式の単純化
        '''
        _t = sp.Symbol('t')
        return (self.p * sp.exp(_t) + self.q) ** self.n

@param('lmd', 'λ', min_value=0, min_include=False)
class PoissonDistribution(DiscreteProbabilityDistribution):

    param_validity = {}
    _lmd, _x = sp.symbols('λ x')
    _PDF = sp.exp(-_lmd) * _lmd ** _x / sp.factorial(_x)

    def __init__(self, lmd: Any=sp.Symbol('λ')) -> None:
        super().__init__()
        self.lmd = lmd

    @property
    def mgf(self) -> Any:
        ''' 式の単純化
        '''
        _t = sp.Symbol('t')
        return sp.exp(self.lmd * (sp.exp(_t) - 1))

@param('k', 'k', min_value=0, min_include=False)
@param_p_q
class NegativeBinomialDistribution(DiscreteProbabilityDistribution):

    param_validity = {}
    _k, _p, _q, _x = sp.symbols('k p q x')
    _PDF = sp.binomial(_k + _x - 1, _x) * _p ** _k * _q ** (_x)

    def __init__(self, k: Any=sp.Symbol('k'),  p: Any=sp.Symbol('p')) -> None:
        super().__init__()
        self.k = k
        self.p = p

    @property
    def mgf(self) -> Any:
        ''' 式の単純化
        '''
        _t = sp.Symbol('t')
        return self.p ** self.k * (1 - self.q * sp.exp(_t)) ** (-self.k)

@param_as_min_max('a', 'a', 'x_min', param_type=int)
@param_as_min_max('b', 'b', 'x_max', param_type=int)
class UniformDistributionOfDiscreteType(DiscreteProbabilityDistribution):

    param_validity = {}
    _a, _b = sp.symbols('a b')
    _PDF = 1 / (_b - _a + 1)

    def __init__(self, a: Any=sp.Symbol('a'), b: Any=sp.Symbol('b')) -> None:
        super().__init__()
        self.a = a
        self.b = b

    @arg_check(arg_type=int)
    def prob(self, x: int) -> Any:
        return self.pdf if self.x_min <= x <= self.x_max else 0

@param('mu', 'μ')
@param('sigma', 'σ', min_value=0, min_include=False)
class NormalDistribution(ContinuousProbabilityDistribution):

    param_validity = {}
    _mu, _sigma, _x = sp.symbols('μ σ x')
    _PDF = sp.exp(-(_x - _mu) ** 2 / (2 * _sigma ** 2)) / (sp.sqrt(2 * sp.pi) * _sigma)

    def __init__(self, mu: Any=sp.Symbol('μ'), sigma: Any=sp.Symbol('σ')) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    @property
    def mgf(self) -> Any:
        ''' 式の単純化
        '''
        _t = sp.Symbol('t')
        return sp.exp(self.mu * _t + self.sigma ** 2 * _t ** 2 / 2)

@param('alpha', 'α', min_value=0, min_include=False)
@param('lmd', 'λ', min_value=0, min_include=False)
class GammaDistribution(ContinuousProbabilityDistribution):

    param_validity = {}
    x_min = 0
    _alpha, _lmd, _x = sp.symbols('α λ x')
    _PDF = _lmd ** _alpha * _x ** (_alpha - 1) * sp.exp(-_lmd * _x) / sp.gamma(_alpha)

    def __init__(self, alpha: Any=sp.Symbol('α'), lmd: Any=sp.Symbol('λ')) -> None:
        super().__init__()
        self.alpha = alpha
        self.lmd = lmd

    @property
    def cdf(self) -> Any:
        ''' 計算高速化のため再定義
        '''
        _z = sp.Symbol('z')
        return sp.lowergamma(self.alpha, self.lmd * _z) / sp.gamma(self.alpha)

@param('alpha', 'α', min_value=0, min_include=False)
@param('beta', 'β', min_value=0, min_include=False)
class BetaDistribution(ContinuousProbabilityDistribution):

    param_validity = {}
    x_min = 0
    x_max = 1
    _alpha, _beta, _x = sp.symbols('α β x')
    _PDF = _x ** (_alpha - 1) * (1 - _x) ** (_beta - 1) / sp.beta(_alpha, _beta)

    def __init__(self, alpha: Any=sp.Symbol('α'), beta: Any=sp.Symbol('β')) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

@param('alpha', 'α', min_value=0, min_include=False)
@param('lmd', 'λ')
class CauchyDistribution(ContinuousProbabilityDistribution):

    param_validity = {}
    _alpha, _lmd, _x = sp.symbols('α λ x')
    _PDF = _alpha / (sp.pi * (_alpha ** 2 + (_x - _lmd) ** 2))

    def __init__(self, alpha: Any=sp.Symbol('α'), lmd: Any=sp.Symbol('λ')) -> None:
        super().__init__()
        self.alpha = alpha
        self.lmd = lmd

@param('mu', 'μ')
@param('sigma', 'σ', min_value=0, min_include=False)
class LogNormalDistribution(ContinuousProbabilityDistribution):

    param_validity = {}
    x_min = 0
    _mu, _sigma, _x = sp.symbols('μ σ x')
    _PDF = sp.exp(-(sp.log(_x) - _mu) ** 2 / (2 * _sigma ** 2)) / (sp.sqrt(2 * sp.pi) * _sigma * _x)

    def __init__(self, mu: Any=sp.Symbol('μ'), sigma: Any=sp.Symbol('σ')) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma

@param('a', 'a', min_value=0, min_include=False)
@param_as_min_max('x0', 'x0', 'x_min', min_value=0, min_include=False)
class ParetoDistribution(ContinuousProbabilityDistribution):

    param_validity = {}
    _a, _x0, _x = sp.symbols('a x0 x')
    _PDF = (_a / _x0) * (_x0 / _x) ** (_a + 1)

    def __init__(self, a: Any=sp.Symbol('a'), x0: Any=sp.Symbol('x0')) -> None:
        super().__init__()
        self.a = a
        self.x0 = x0

@param('a', 'a', min_value=0, min_include=False)
@param('b', 'b', min_value=0, min_include=False)
class WeibulDistribution(ContinuousProbabilityDistribution):

    param_validity = {}
    x_min = 0
    _a, _b, _x = sp.symbols('a b x')
    _PDF = (_b * _x ** (_b - 1) / _a ** _b) * sp.exp(-(_x / _a) ** _b)

    def __init__(self, a: Any=sp.Symbol('a'), b: Any=sp.Symbol('b')) -> None:
        super().__init__()
        self.a = a
        self.b = b

@param('k', 'k', min_value=1)
class ChiSquaredDistribution(ContinuousProbabilityDistribution):

    param_validity = {}
    x_min = 0
    _k, _x = sp.symbols('k x')
    _PDF = _x ** (_k / 2 - 1) * sp.exp(-_x / 2) / (2 ** (_k / 2) * sp.gamma(_k / 2))

    def __init__(self, k: Any=sp.Symbol('k')) -> None:
        super().__init__()
        self.k = k

    @property
    def cdf(self) -> Any:
        ''' 計算高速化のため再定義
        '''
        _z = sp.Symbol('z')
        return sp.lowergamma(self.k / 2, _z / 2) / sp.gamma(self.k / 2)

class CounterDict(Dict[float, float]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def mean(self) -> float:
        total_num = sum(self.values())
        total = 0
        for key, value in self.items():
            total += key * value
        return total / total_num

    def var(self, ddof: int=0) -> float:
        total_num = sum(self.values()) - ddof
        ex = self.mean()
        total = 0
        for key, value in self.items():
            total += (key - ex) ** 2 * value
        return total / total_num

    def __pow__(self, num: int) -> Dict[float, float]:
        return CounterDict([(key ** num, value)
                            for key, value in self.items()])

class ParameterNotGivenError(Exception):
    pass