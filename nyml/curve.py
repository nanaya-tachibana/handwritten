# -*- coding: utf-8 -*-
from math import *
import numpy as np

def generate_curve_samples(func, x_range, num=100):
    """Return a matrix which first col is x values and secord col is funcion values.

    Parameters
    ------
    func: string
    A string function

    x_range: tuple
    Dot sample range

    num: integer
    Sample size
    """
    x = np.linspace(*x_range, num=num)
    y = np.array([eval(func) for x in x])
    return np.vstack([x,y]).T
