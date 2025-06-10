# -*- coding: utf-8 -*-

import numpy as np
import gmpy2 as gp


sin = np.vectorize(gp.sin, otypes=(object,))
cos = np.vectorize(gp.cos, otypes=(object,))
tan = np.vectorize(gp.tan, otypes=(object,))
sqrt = np.vectorize(gp.sqrt, otypes=(object,))
exp = np.vectorize(gp.exp, otypes=(object,))
gamma = np.vectorize(gp.gamma, otypes=(object,))
lngamma = np.vectorize(gp.lngamma, otypes=(object,))

