"""Functions for linear algebra
"""


import numpy as np
from scipy import linalg
import flamp, gmpy2

from typing import Tuple


class LinSysSolver:
    
    def __init__(self, name: str = "lin_solver", dependencies: list = []) -> None:
        self._name = name
        self._dependencies = dependencies
    
    def eig(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def eigh(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def eig_g(self, A: np.ndarray, B: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def inv(self, A: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError


class StdLinSolver(LinSysSolver):
    
    def __init__(self) -> None:
        super().__init__(
            name="standard_solver", 
            dependencies=["numpy", "scipy"])
    
    def eig(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return np.linalg.eig(A)
    
    def eigh(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(A, **kwargs)
    
    def eig_g(self, A: np.ndarray, B: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        return linalg.eig(A, b=B, **kwargs)
    
    def inv(self, A: np.ndarray, **kwargs) -> np.ndarray:
        return np.linalg.inv(A)


class MultiPrecLinSolver(LinSysSolver):
    
    def __init__(self, prec: int = 113) -> None:
        super().__init__(
            name="multiprec_solver", 
            dependencies=["gmpy2",])
        self._prec = prec
    
    def eig(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        with gmpy2.local_context(gmpy2.context(), precision=self._prec):
            return flamp.eig(A, **kwargs)
    
    def eigh(self, A: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        with gmpy2.local_context(gmpy2.context(), precision=self._prec):
            return flamp.eigh(A, **kwargs)
    
    def inv(self, A: np.ndarray, **kwargs) -> np.ndarray:
        with gmpy2.local_context(gmpy2.context(), precision=self._prec):
            return flamp.inverse(A)


def eig_generalized(M: np.ndarray, K: np.ndarray, diag: bool = False, 
    solver: LinSysSolver = StdLinSolver(), **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Solve generalized eigenvalue problem.
    This should serve as the final interface for generalized eigenproblems.
    """
    
    if diag:
        A = (K.T / np.diag(M)).T
    else:
        A = solver.inv(M) @ K
    
    return solver.eig(A, **kwargs)

